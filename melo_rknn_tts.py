#!/usr/bin/env python3
"""Hybrid MeloTTS: ONNX encoder (CPU) + RKNN decoder (NPU)."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

_LOGGER = logging.getLogger("melo_rknn_tts")

MELO_PKG = Path(__file__).resolve().parent / "melo_rknn"
DEC_LEN = 65536 // 512  # 128 frames


def _ensure_melo_path() -> Path:
    if str(MELO_PKG) not in sys.path:
        sys.path.insert(0, str(MELO_PKG))
    return MELO_PKG


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
        "decoder.rknn",
    ):
        p = model_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No Melo decoder under {model_dir}")


def npu_load() -> str:
    try:
        return Path("/sys/kernel/debug/rknpu/load").read_text().strip()
    except OSError:
        return "n/a"


class MeloRknnTts:
    """Keep Melo encoder+decoder loaded for repeated requests."""

    def __init__(
        self,
        model_dir: Path,
        speed: float = 0.8,
        sample_rate: int = 44100,
        decoder_path: Optional[Path] = None,
        core_mask: Optional[int] = None,
    ):
        _ensure_melo_path()
        from utils import Lexicon, intersperse, split_sentences_zh  # type: ignore
        import melotts_rknn as melo  # type: ignore
        from rknn_lite_ctypes import RKNNLite  # type: ignore

        self.model_dir = Path(model_dir)
        self.speed = float(speed)
        self.sample_rate = int(sample_rate)
        self.decoder_path = Path(decoder_path) if decoder_path else pick_decoder(self.model_dir)
        self._melo = melo
        self._split = split_sentences_zh
        self._intersperse = intersperse
        self._lexicon = Lexicon(
            str(self.model_dir / "lexicon.txt"),
            str(self.model_dir / "tokens.txt"),
        )

        cwd = Path.cwd()
        os.chdir(self.model_dir)
        try:
            t0 = time.perf_counter()
            self._enc = ort.InferenceSession(
                str(self.model_dir / "encoder.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=ort.SessionOptions(),
            )
            self._dec = RKNNLite()
            ret = self._dec.load_rknn(str(self.decoder_path))
            if ret != 0:
                raise RuntimeError(f"load decoder failed: {ret}")
            if core_mask is None:
                soc = detect_soc()
                core_mask = (
                    int(RKNNLite.NPU_CORE_0_1_2) if "3588" in soc else int(RKNNLite.NPU_CORE_AUTO)
                )
            self.core_mask = int(core_mask)
            ret = self._dec.init_runtime(core_mask=self.core_mask)
            if ret != 0:
                raise RuntimeError(f"init_runtime failed: {ret}")
            self._g = np.fromfile(str(self.model_dir / "g.bin"), dtype=np.float32).reshape(
                1, 256, 1
            )
            self._dec_len = DEC_LEN
            _LOGGER.info(
                "Melo RKNN ready in %.0fms decoder=%s core_mask=%s",
                (time.perf_counter() - t0) * 1000,
                self.decoder_path.name,
                self.core_mask,
            )
        finally:
            os.chdir(cwd)

    def synthesize(self, text: str) -> Tuple[np.ndarray, dict]:
        m = self._melo
        enc_ms = 0.0
        dec_ms = 0.0
        audio_list = []
        cwd = Path.cwd()
        os.chdir(self.model_dir)
        try:
            for se in self._split(text):
                phone_str, yinjie_num, phones, tones = self._lexicon.convert(se)
                phone_str = self._intersperse(phone_str, 0)
                phones = np.array(self._intersperse(phones, 0), dtype=np.int32)
                tones = np.array(self._intersperse(tones, 0), dtype=np.int32)
                yinjie_num = np.array(yinjie_num, dtype=np.int32) * 2
                yinjie_num[0] += 1
                pron_slices = m.generate_pronounce_slice(yinjie_num)
                phone_len = phones.shape[-1]
                language = np.array([3] * phone_len, dtype=np.int32)

                t0 = time.perf_counter()
                z_p, pronoun_lens, audio_len = self._enc.run(
                    None,
                    input_feed={
                        "phone": phones,
                        "g": self._g,
                        "tone": tones,
                        "language": language,
                        "noise_scale": np.array([0], dtype=np.float32),
                        "length_scale": np.array([1.0 / self.speed], dtype=np.float32),
                        "noise_scale_w": np.array([0], dtype=np.float32),
                        "sdp_ratio": np.array([0], dtype=np.float32),
                    },
                )
                enc_ms += (time.perf_counter() - t0) * 1000

                audio_len = int(audio_len[0])
                actual_size = z_p.shape[-1]
                dec_slice_num = int(np.ceil(actual_size / self._dec_len))
                z_p = np.pad(
                    z_p,
                    pad_width=(
                        (0, 0),
                        (0, 0),
                        (0, dec_slice_num * self._dec_len - actual_size),
                    ),
                    mode="constant",
                    constant_values=0,
                )
                pron_num = m.generate_word_pron_num(pronoun_lens, pron_slices)
                sub_audio_list = []
                pron_num_slices, zp_slices, strip_flags, _pron_lens, is_long = (
                    m.generate_decode_slices(pron_num, self._dec_len)
                )

                for i in range(len(pron_num_slices)):
                    pron_start, pron_end = pron_num_slices[i]
                    zp_start, zp_end = zp_slices[i]
                    if is_long[i]:
                        t1 = time.perf_counter()
                        parts = m.decode_long_word(
                            self._dec, z_p[..., zp_start:zp_end], self._g, self._dec_len
                        )
                        dec_ms += (time.perf_counter() - t1) * 1000
                        sub_audio_list.extend(parts)
                    else:
                        sub_dec_len = zp_end - zp_start
                        sub_audio_len = 512 * sub_dec_len
                        zp_slice = z_p[..., zp_start:zp_end]
                        if zp_slice.shape[-1] < self._dec_len:
                            zp_slice = np.concatenate(
                                (
                                    zp_slice,
                                    np.zeros(
                                        (
                                            *zp_slice.shape[:-1],
                                            self._dec_len - zp_slice.shape[-1],
                                        ),
                                        dtype=np.float32,
                                    ),
                                ),
                                axis=-1,
                            )
                        t1 = time.perf_counter()
                        audio = self._dec.inference(inputs=[zp_slice, self._g])[0].flatten()
                        dec_ms += (time.perf_counter() - t1) * 1000
                        audio = audio[:sub_audio_len]
                        if strip_flags[i][0]:
                            audio = audio[512 * pron_num[pron_start] :]
                        if strip_flags[i][1]:
                            audio = audio[: -512 * pron_num[pron_end - 1]]
                        sub_audio_list.append(audio)

                sub_audio = m.merge_sub_audio(sub_audio_list, 0, audio_len)
                audio_list.append(sub_audio)

            audio = m.audio_numpy_concat(audio_list, sr=self.sample_rate, speed=self.speed)
        finally:
            os.chdir(cwd)

        dur = float(len(audio) / self.sample_rate) if self.sample_rate else 0.0
        total_ms = enc_ms + dec_ms
        stats = {
            "audio_sec": dur,
            "encoder_ms": enc_ms,
            "decoder_ms": dec_ms,
            "rtf": (total_ms / 1000.0) / dur if dur > 0 else 0.0,
            "npu": npu_load(),
            "core_mask": self.core_mask,
        }
        return np.asarray(audio, dtype=np.float32), stats
