#!/usr/bin/env python3
"""Hybrid Kokoro TTS: encoder+har ONNX (CPU) + decoder RKNN (NPU).

Artefacts match marty1885/kokoro-server (StyleTTS2 + iSTFTNet split):
  encoder.onnx / har_generator.onnx / decoder_rk3588.rknn / voices/*.npy / config.json
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

_LOGGER = logging.getLogger("kokoro_rknn_tts")

SR = 24000
SAMPLES_PER_FR = 600
T_FIX_DEFAULT = 50
DEPOP_CTX = 2
DEPOP_OVL = 480


def detect_soc() -> str:
    env = (os.environ.get("RK_SOC") or "").strip().lower()
    if env:
        return env
    try:
        compat = Path("/proc/device-tree/compatible").read_bytes().decode("utf-8", "ignore")
    except OSError:
        compat = ""
    if "rk3588" in compat:
        return "rk3588"
    if "rk3566" in compat or "rk3568" in compat:
        return "rk3566"
    return "rk3566"


def pick_decoder(model_dir: Path, soc: Optional[str] = None) -> Path:
    soc = (soc or detect_soc()).lower()
    preferred = "decoder_rk3588.rknn" if "3588" in soc else "decoder_rk3566.rknn"
    for name in (
        preferred,
        "decoder_rk3566.rknn",
        "decoder_rk3588.rknn",
        "kokoro_decoder.rknn",
        "decoder.rknn",
    ):
        p = model_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No Kokoro decoder .rknn under {model_dir}")


def npu_load() -> str:
    try:
        return Path("/sys/kernel/debug/rknpu/load").read_text().strip()
    except OSError:
        return "n/a"


def depop_append(buf: np.ndarray, chunk: np.ndarray, blend: int = DEPOP_OVL) -> np.ndarray:
    if buf.size < blend or chunk.size < blend:
        return np.concatenate([buf, chunk])
    w = np.linspace(0.0, 1.0, blend, endpoint=False, dtype=np.float32)
    blended = buf[-blend:] * (1.0 - w) + chunk[:blend] * w
    return np.concatenate([buf[:-blend], blended, chunk[blend:]])


class IStft:
    def __init__(self, n_fft: int = 20, hop: int = 5):
        self.n_fft, self.hop = n_fft, hop
        n = np.arange(n_fft)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)
        self.win_sq = (win * win).astype(np.float32)
        inv_window = (win / n_fft).astype(np.float32)
        F = n_fft // 2 + 1
        k = np.arange(F)
        dbl = np.ones(F, np.float32)
        dbl[1 : F - 1] = 2.0
        angle = 2 * np.pi * np.outer(n, k) / n_fft
        self.cos_w = (np.cos(angle).T * inv_window * dbl[:, None]).astype(np.float32)
        self.sin_w = (np.sin(angle).T * inv_window * dbl[:, None]).astype(np.float32)
        self.n_bins = F

    def __call__(self, spec, phase):
        real = spec * np.cos(phase)
        imag = spec * np.sin(phase)
        fr_r = np.einsum("bft,fn->btn", real, self.cos_w)
        fr_i = np.einsum("bft,fn->btn", imag, self.sin_w)
        frames = fr_r - fr_i
        B, T_frames, _ = frames.shape
        ratio = self.n_fft // self.hop
        out_len = (T_frames - 1) * self.hop + self.n_fft
        out = np.zeros((B, out_len), np.float32)
        flat_len = T_frames * self.hop
        for k in range(ratio):
            contrib = frames[:, :, k * self.hop : (k + 1) * self.hop].reshape(B, flat_len)
            out[:, k * self.hop : k * self.hop + flat_len] += contrib
        env = np.zeros(out_len, np.float32)
        for t in range(T_frames):
            env[t * self.hop : t * self.hop + self.n_fft] += self.win_sq
        out /= np.maximum(env, 1e-11)
        pad = self.n_fft // 2
        return out[:, pad:-pad] if pad else out


def _reshape_nchw(flat: np.ndarray, dims: Tuple[int, ...]) -> np.ndarray:
    shape = tuple(int(d) for d in dims if int(d) > 0)
    if not shape:
        return flat
    n = int(np.prod(shape))
    if flat.size == n:
        return flat.reshape(shape)
    if flat.size > n and flat.size % n == 0:
        return flat[:n].reshape(shape)
    return flat.reshape((-1, *shape[1:])) if len(shape) > 1 else flat


def _ensure_melo_path() -> None:
    pkg = Path(__file__).resolve().parent / "melo_rknn"
    if str(pkg.parent) not in sys.path:
        sys.path.insert(0, str(pkg.parent))


class RknnDecoder:
    def __init__(self, rknn_path: Path, core_mask: Optional[int] = None):
        _ensure_melo_path()
        from melo_rknn.rknn_lite_ctypes import RKNNLite

        self.r = RKNNLite()
        if self.r.load_rknn(str(rknn_path)) != 0:
            raise RuntimeError(f"load_rknn failed: {rknn_path}")
        soc = detect_soc()
        if core_mask is None:
            core_mask = int(RKNNLite.NPU_CORE_0_1_2) if "3588" in soc else int(RKNNLite.NPU_CORE_AUTO)
        self.core_mask = int(core_mask)
        if self.r.init_runtime(core_mask=self.core_mask) != 0:
            raise RuntimeError("init_runtime failed")
        self._out_dims = []
        for attr in getattr(self.r, "_out_attrs", []) or []:
            n = int(getattr(attr, "n_dims", 0) or 0)
            self._out_dims.append(tuple(int(attr.dims[i]) for i in range(n)))

    def __call__(self, asr, F0, N, s, har):
        outs = self.r.inference(inputs=[asr, F0, N, s, har], data_format=["nchw"] * 5)
        shaped = []
        for i, arr in enumerate(outs):
            if i < len(self._out_dims) and self._out_dims[i]:
                shaped.append(_reshape_nchw(arr, self._out_dims[i]))
            else:
                # Fallback: spec/phase are (1, 11, T) for gen_istft_n_fft=20
                bins = 11
                if arr.size % bins == 0:
                    t = arr.size // bins
                    shaped.append(arr.reshape(1, bins, t))
                else:
                    shaped.append(arr)
        return shaped[0], shaped[1]

    def close(self):
        self.r.release()


def load_vocab(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("vocab") or {}


def phonemes_to_ids(phonemes: str, vocab: dict) -> list[int]:
    ids = [vocab[p] for p in phonemes if p in vocab]
    missing = [p for p in phonemes if p not in vocab]
    if missing:
        _LOGGER.warning("dropped %s chars not in vocab: %s", len(missing), missing[:8])
    return ids


def _punct(ch: str) -> str:
    return {"，": ",", "。": ".", "！": "!", "？": "?", "、": ",", "；": ";", "：": ":"}.get(ch, ch)


_DIGIT_ZH = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def _an2cn_int(n: int) -> str:
    try:
        import cn2an

        return cn2an.an2cn(str(n))
    except Exception:
        # Minimal fallback for 0..59
        if n <= 10:
            return _DIGIT_ZH[str(n)] if n < 10 else "十"
        if n < 20:
            return "十" + _DIGIT_ZH[str(n - 10)]
        tens, ones = divmod(n, 10)
        return _DIGIT_ZH[str(tens)] + "十" + (_DIGIT_ZH[str(ones)] if ones else "")


def _expand_clock(match: re.Match) -> str:
    """3:15 → 三点十五分 so colon cannot glue 三+十五 into 三十五."""
    hour, minute = int(match.group(1)), int(match.group(2))
    h = _an2cn_int(hour)
    if minute == 0:
        return f"{h}点"
    if minute < 10:
        return f"{h}点零{_an2cn_int(minute)}分"
    return f"{h}点{_an2cn_int(minute)}分"


def _normalize_zh_tts_text(text: str) -> str:
    """Fix common HA/serial patterns before G2P.

    - ``3:15`` must become 三点十五分 (bare 三:十五 sounds like 三十五).
    - ``No.001`` / leading-zero codes → 编号零零一.
    """
    # Clock times first (before leading-zero rewrite eats :05 / :00).
    text = re.sub(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", _expand_clock, text)
    # Abbreviation No./NO. before digits → 编号
    text = re.sub(r"(?i)(?<![A-Za-z])no\.?\s*(?=\d)", "编号", text)
    text = re.sub(r"编号\s*编号", "编号", text)
    # Codes with leading zeros (IDs like 001, 007) → 零零一, not 一.
    text = re.sub(
        r"(?<!\d)0\d+(?!\d)",
        lambda m: "".join(_DIGIT_ZH[c] for c in m.group(0)),
        text,
    )
    return text


_ZHG2P = None


def _en_espeak(text: str) -> str:
    """English → IPA via espeak-ng (already in the RKNN image; no spacy)."""
    import subprocess

    out = subprocess.check_output(
        ["espeak-ng", "-v", "en-us", "-q", "--ipa=3", text],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    # Strip zero-width joiners that espeak sometimes inserts in diphthongs.
    return out.strip().replace("\u200d", "")


def _zhg2p():
    global _ZHG2P
    if _ZHG2P is None:
        from misaki.zh import ZHG2P

        _ZHG2P = ZHG2P(version="1.1", en_callable=_en_espeak)
    return _ZHG2P


def text_to_phonemes(text: str) -> str:
    """Kokoro v1.1-zh G2P: misaki ZHG2P + espeak for Latin; pypinyin fallback."""
    os.environ.setdefault("PYTORCH_MATCHER_LOGLEVEL", "WARNING")
    text = _normalize_zh_tts_text(text)
    try:
        ph, _ = _zhg2p()(text)
        return ph or text
    except Exception as e:
        _LOGGER.warning("misaki ZHG2P unavailable (%s)", e)
    try:
        from pypinyin import Style, lazy_pinyin

        out = []
        for ch in text:
            mapped = _punct(ch)
            if "\u4e00" <= ch <= "\u9fff":
                py = lazy_pinyin(ch, style=Style.TONE3)
                out.append(py[0] if py else mapped)
            else:
                out.append(mapped)
        return "".join(out)
    except Exception:
        return "".join(_punct(c) for c in text)


def load_voice(model_dir: Path, voice: str, n_phonemes: int) -> np.ndarray:
    path = Path(voice)
    if not path.is_file():
        for folder in ("voices", "voices_npy"):
            cand = model_dir / folder / f"{Path(voice).stem}.npy"
            if cand.is_file():
                path = cand
                break
    if not path.is_file():
        voices = sorted((model_dir / "voices").glob("*.npy")) + sorted(
            (model_dir / "voices_npy").glob("*.npy")
        )
        if not voices:
            raise FileNotFoundError(f"no Kokoro voices under {model_dir}")
        path = voices[0]
        _LOGGER.warning("voice %s missing, using %s", voice, path.name)
    pack = np.load(path)
    if pack.ndim == 3:
        pack = pack[max(0, min(pack.shape[0] - 1, n_phonemes - 1))]
    return pack.astype(np.float32).reshape(1, 256)


class KokoroRknnTts:
    def __init__(
        self,
        model_dir: Path,
        voice: str = "zf_001",
        speed: float = 1.0,
        t_fix: int = T_FIX_DEFAULT,
        core_mask: Optional[int] = None,
    ):
        self.model_dir = Path(model_dir)
        self.voice = voice
        self.speed = float(speed)
        self.sample_rate = SR
        self.t_fix = int(t_fix)
        enc = self.model_dir / "encoder.onnx"
        har = self.model_dir / "har_generator.onnx"
        if not enc.is_file():
            enc = self.model_dir / "kokoro_encoder.onnx"
        if not har.is_file():
            har = self.model_dir / "har_generator.onnx"
        cfg = self.model_dir / "config.json"
        if not cfg.is_file():
            raise FileNotFoundError(f"Kokoro config.json missing in {self.model_dir}")
        self.vocab = load_vocab(cfg)
        self.decoder_path = pick_decoder(self.model_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("TTS_THREAD_NUM", "2") or 2)
        t0 = time.perf_counter()
        self.enc = ort.InferenceSession(str(enc), sess_options=so, providers=["CPUExecutionProvider"])
        self.har = ort.InferenceSession(str(har), sess_options=so, providers=["CPUExecutionProvider"])
        self.dec = RknnDecoder(self.decoder_path, core_mask=core_mask)
        self.istft = IStft()
        # Warm NPU
        T = self.t_fix
        _ = self.dec(
            np.zeros((1, 512, T), np.float32),
            np.zeros((1, 2 * T), np.float32),
            np.zeros((1, 2 * T), np.float32),
            np.zeros((1, 128), np.float32),
            np.zeros((1, 22, 2 * T * 300 // 5 + 1), np.float32),
        )
        _LOGGER.info(
            "Kokoro RKNN ready in %.0fms decoder=%s voice=%s core_mask=%s",
            (time.perf_counter() - t0) * 1000,
            self.decoder_path.name,
            self.voice,
            self.dec.core_mask,
        )

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        phonemes: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        phonemes = phonemes or text_to_phonemes(text)
        ids = phonemes_to_ids(phonemes, self.vocab)
        if not ids:
            raise ValueError(f"no valid phonemes for {text!r} → {phonemes!r}")
        input_ids = np.array([[0, *ids, 0]], np.int64)
        ref_s = load_voice(self.model_dir, voice or self.voice, len(ids))
        speed = np.array([self.speed], np.float32)

        t0 = time.perf_counter()
        asr, F0, N, s, pred_dur = self.enc.run(
            None, {"input_ids": input_ids, "ref_s": ref_s, "speed": speed}
        )
        t_enc = time.perf_counter() - t0
        T = int(asr.shape[2])
        trail = int(np.asarray(pred_dur).reshape(-1)[-1])
        T_emit = max(1, T - trail)

        audio_full = np.empty(0, np.float32)
        t_har = t_dec = t_ist = 0.0
        n_win = 0
        CTX = DEPOP_CTX
        T_FIX = self.t_fix
        STEP = T_FIX - 2 * CTX
        emit_offset = CTX * SAMPLES_PER_FR
        HALF = DEPOP_OVL // 2
        for emit_start in range(0, T_emit, STEP):
            emit_end = min(emit_start + STEP, T_emit)
            win_start, win_end = emit_start - CTX, emit_end + CTX
            a_lo, a_hi = max(0, win_start), min(T, win_end)
            lpad = a_lo - win_start
            rpad = win_end - a_hi
            rpad += T_FIX - (a_hi - a_lo + lpad + rpad)
            a = asr[:, :, a_lo:a_hi]
            f = F0[:, 2 * a_lo : 2 * a_hi]
            n = N[:, 2 * a_lo : 2 * a_hi]
            if lpad or rpad:
                a = np.pad(a, ((0, 0), (0, 0), (lpad, rpad)))
                f = np.pad(f, ((0, 0), (2 * lpad, 2 * rpad)))
                n = np.pad(n, ((0, 0), (2 * lpad, 2 * rpad)))
            a = np.ascontiguousarray(a, np.float32)
            f = np.ascontiguousarray(f, np.float32)
            n = np.ascontiguousarray(n, np.float32)

            t1 = time.perf_counter()
            har = self.har.run(None, {"F0": f})[0].astype(np.float32)
            t_har += time.perf_counter() - t1
            t1 = time.perf_counter()
            spec, phase = self.dec(a, f, n, s, har)
            t_dec += time.perf_counter() - t1
            t1 = time.perf_counter()
            audio = self.istft(spec, phase)
            t_ist += time.perf_counter() - t1

            emit_samples = (emit_end - emit_start) * SAMPLES_PER_FR
            left_ext = HALF if emit_start > 0 else 0
            right_ext = HALF if emit_end < T_emit else 0
            piece = audio.reshape(-1)[
                emit_offset - left_ext : emit_offset + emit_samples + right_ext
            ].astype(np.float32, copy=False)
            audio_full = depop_append(audio_full, piece)
            n_win += 1

        dur = float(audio_full.size / SR) if SR else 0.0
        total = t_enc + t_har + t_dec + t_ist
        stats = {
            "audio_sec": dur,
            "encoder_ms": t_enc * 1000,
            "har_ms": t_har * 1000,
            "decoder_ms": t_dec * 1000,
            "istft_ms": t_ist * 1000,
            "windows": n_win,
            "phonemes": phonemes,
            "rtf": (total / dur) if dur > 0 else 0.0,
            "npu": npu_load(),
            "core_mask": self.dec.core_mask,
        }
        return audio_full, stats


def _write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    import wave

    pcm = np.clip(audio, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", default="你好，今天天气不错。")
    ap.add_argument("--phonemes")
    ap.add_argument("--voice", default="zf_001")
    ap.add_argument("--out", default="/tmp/kokoro-npu.wav")
    args = ap.parse_args()
    tts = KokoroRknnTts(Path(args.model_dir), voice=args.voice)
    audio, stats = tts.synthesize(args.text, voice=args.voice, phonemes=args.phonemes)
    _write_wav(Path(args.out), audio)
    print(stats)
    print("wrote", args.out, "samples", audio.size, "peak", float(np.max(np.abs(audio))))
