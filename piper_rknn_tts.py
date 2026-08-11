#!/usr/bin/env python3
"""Hybrid Piper TTS: ONNX encoder (CPU) + RKNN decoder (NPU)."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import subprocess
import time
import wave
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

BOS, EOS, PAD = "^", "$", "_"
HOP = 256  # Piper VITS hop length
DECODER_FRAMES = 150  # fixed RKNN time dim


# -------- RKNN decoder (ctypes) --------
class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RknnDecoder:
    RKNN_TENSOR_FLOAT32 = 0
    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1

    def __init__(self, model_path: Path):
        self.lib = ctypes.CDLL("librknnrt.so")
        self.lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.lib.rknn_init.restype = ctypes.c_int
        for name in (
            "rknn_query",
            "rknn_inputs_set",
            "rknn_run",
            "rknn_outputs_get",
            "rknn_outputs_release",
            "rknn_destroy",
        ):
            getattr(self.lib, name).restype = ctypes.c_int
        self.lib.rknn_query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.lib.rknn_inputs_set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RKNNInput),
        ]
        self.lib.rknn_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.rknn_outputs_get.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RKNNOutput),
            ctypes.c_void_p,
        ]
        self.lib.rknn_outputs_release.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RKNNOutput),
        ]
        self.lib.rknn_destroy.argtypes = [ctypes.c_void_p]

        data = Path(model_path).read_bytes()
        self._buf = ctypes.create_string_buffer(data, len(data))
        self.ctx = ctypes.c_void_p()
        ret = self.lib.rknn_init(
            ctypes.byref(self.ctx), self._buf, len(data), 0, None
        )
        if ret != 0:
            raise RuntimeError(f"rknn_init failed: {ret}")

        io = RKNNInputOutputNum()
        ret = self.lib.rknn_query(
            self.ctx, self.RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io), ctypes.sizeof(io)
        )
        if ret != 0:
            raise RuntimeError(f"query io failed: {ret}")
        self.n_input = io.n_input
        self.n_output = io.n_output
        self.attrs = []
        for i in range(self.n_input):
            a = RKNNTensorAttr()
            a.index = i
            ret = self.lib.rknn_query(
                self.ctx,
                self.RKNN_QUERY_INPUT_ATTR,
                ctypes.byref(a),
                ctypes.sizeof(a),
            )
            if ret != 0:
                raise RuntimeError(f"input attr {i} failed: {ret}")
            self.attrs.append(a)

    def run(self, z: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
        """z: [1,192,T], y_mask: [1,1,T] float32; T must be DECODER_FRAMES."""
        assert z.shape == (1, 192, DECODER_FRAMES), z.shape
        assert y_mask.shape == (1, 1, DECODER_FRAMES), y_mask.shape
        z = np.ascontiguousarray(z, dtype=np.float32)
        y_mask = np.ascontiguousarray(y_mask, dtype=np.float32)
        arrays = [
            np.ascontiguousarray(z.reshape(-1), dtype=np.float32),
            np.ascontiguousarray(y_mask.reshape(-1), dtype=np.float32),
        ]
        inputs = (RKNNInput * self.n_input)()
        for i, arr in enumerate(arrays):
            inputs[i].index = i
            inputs[i].buf = arr.ctypes.data_as(ctypes.c_void_p)
            inputs[i].size = arr.nbytes
            inputs[i].pass_through = 0
            inputs[i].type = self.RKNN_TENSOR_FLOAT32
            inputs[i].fmt = self.attrs[i].fmt
        # keep arrays alive across rknn calls
        self._last_inputs = arrays
        ret = self.lib.rknn_inputs_set(self.ctx, self.n_input, inputs)
        if ret != 0:
            raise RuntimeError(f"inputs_set failed: {ret}")
        ret = self.lib.rknn_run(self.ctx, None)
        if ret != 0:
            raise RuntimeError(f"run failed: {ret}")
        outs = (RKNNOutput * self.n_output)()
        for i in range(self.n_output):
            outs[i].want_float = 1
        ret = self.lib.rknn_outputs_get(self.ctx, self.n_output, outs, None)
        if ret != 0:
            raise RuntimeError(f"outputs_get failed: {ret}")
        # output [1,1,38400] => 150*256
        n = outs[0].size // 4
        audio = np.ctypeslib.as_array(
            (ctypes.c_float * n).from_address(outs[0].buf)
        ).copy()
        self.lib.rknn_outputs_release(self.ctx, self.n_output, outs)
        return audio

    def close(self) -> None:
        if self.ctx:
            self.lib.rknn_destroy(self.ctx)
            self.ctx = None


# -------- phonemize via espeak-ng IPA --------
def _build_phone_trie(id_map: dict) -> List[str]:
    # longest-first for greedy match
    return sorted(id_map.keys(), key=len, reverse=True)


def espeak_ipa(text: str, voice: str = "cmn") -> str:
    out = subprocess.check_output(
        ["espeak-ng", "-v", voice, "-q", "--ipa=3", text],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    # strip ZWJ used by espeak for ligatures
    return out.replace("\u200d", "").strip()


def ipa_to_phonemes(ipa: str, phones: Sequence[str]) -> List[str]:
    phonemes: List[str] = []
    i = 0
    while i < len(ipa):
        ch = ipa[i]
        if ch in "\n\r":
            i += 1
            continue
        matched = None
        for p in phones:
            if ipa.startswith(p, i):
                matched = p
                break
        if matched is None:
            # skip unknown char
            i += 1
            continue
        phonemes.append(matched)
        i += len(matched)
    return phonemes


def phonemes_to_ids(phonemes: Sequence[str], id_map: dict) -> List[int]:
    ids: List[int] = list(id_map[BOS])
    for ph in phonemes:
        if ph not in id_map:
            continue
        ids.extend(id_map[ph])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids


def audio_float_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def npu_load() -> str:
    p = Path("/sys/kernel/debug/rknpu/load")
    return p.read_text().strip() if p.exists() else "n/a"


class PiperRknnTts:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        with open(model_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.sample_rate = int(self.config["audio"]["sample_rate"])
        self.voice = self.config.get("espeak", {}).get("voice", "cmn")
        self.id_map = self.config["phoneme_id_map"]
        self.phones = _build_phone_trie(self.id_map)
        inf = self.config.get("inference", {})
        self.noise_scale = float(inf.get("noise_scale", 0.667))
        self.length_scale = float(inf.get("length_scale", 1.0))
        self.noise_w = float(inf.get("noise_w", 0.8))

        so = ort.SessionOptions()
        so.inter_op_num_threads = 2
        so.intra_op_num_threads = 2
        self.encoder = ort.InferenceSession(
            str(model_dir / "encoder.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        dec = None
        for name in ("decoder_rk3588.rknn", "decoder_rk3566.rknn", "decoder.rknn"):
            cand = model_dir / name
            if cand.is_file():
                dec = cand
                break
        if dec is None:
            raise FileNotFoundError(f"No Piper decoder .rknn under {model_dir}")
        self.decoder = RknnDecoder(dec)
        self.enc_inputs = {i.name for i in self.encoder.get_inputs()}

    def close(self) -> None:
        self.decoder.close()

    def text_to_ids(self, text: str) -> List[int]:
        ipa = espeak_ipa(text, self.voice)
        phonemes = ipa_to_phonemes(ipa, self.phones)
        return phonemes_to_ids(phonemes, self.id_map)

    def encode(self, phoneme_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        text = np.expand_dims(np.asarray(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.asarray([text.shape[1]], dtype=np.int64)
        scales = np.asarray(
            [self.noise_scale, self.length_scale, self.noise_w], dtype=np.float32
        )
        feeds = {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
        }
        if "sid" in self.enc_inputs:
            feeds["sid"] = np.asarray([0], dtype=np.int64)
        outs = self.encoder.run(None, feeds)
        z, y_mask = outs[0], outs[1]
        return z.astype(np.float32), y_mask.astype(np.float32)

    def decode_chunked(self, z: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
        """Decode full latent with fixed-150 RKNN windows and stitch."""
        n_frames = z.shape[2]
        # Use non-overlapping windows of DECODER_FRAMES; pad last.
        pieces = []
        for start in range(0, n_frames, DECODER_FRAMES):
            end = min(start + DECODER_FRAMES, n_frames)
            t = end - start
            z_pad = np.zeros((1, 192, DECODER_FRAMES), dtype=np.float32)
            y_pad = np.zeros((1, 1, DECODER_FRAMES), dtype=np.float32)
            z_pad[:, :, :t] = z[:, :, start:end]
            y_pad[:, :, :t] = y_mask[:, :, start:end]
            audio = self.decoder.run(z_pad, y_pad)
            pieces.append(audio[: t * HOP])
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def synthesize(self, text: str) -> Tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        ids = self.text_to_ids(text)
        t_ph = time.perf_counter()
        z, y_mask = self.encode(ids)
        t_enc = time.perf_counter()
        audio = self.decode_chunked(z, y_mask)
        t_dec = time.perf_counter()
        dur = len(audio) / self.sample_rate if self.sample_rate else 0.0
        stats = {
            "phonemes": len(ids),
            "frames": int(z.shape[2]),
            "audio_sec": dur,
            "phonemize_ms": (t_ph - t0) * 1000,
            "encoder_ms": (t_enc - t_ph) * 1000,
            "decoder_ms": (t_dec - t_enc) * 1000,
            "total_ms": (t_dec - t0) * 1000,
            "rtf": ((t_dec - t0) / dur) if dur > 0 else None,
            "npu": npu_load(),
        }
        return audio, stats


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = audio_float_to_int16(audio)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/root/models/piper-zh")
    ap.add_argument("--text", default="你好，欢迎使用瑞芯微语音合成。")
    ap.add_argument("--out", default="/tmp/piper_rknn.wav")
    args = ap.parse_args()

    tts = PiperRknnTts(Path(args.model_dir))
    try:
        print("npu before:", npu_load(), flush=True)
        audio, stats = tts.synthesize(args.text)
        write_wav(Path(args.out), audio, tts.sample_rate)
        print("text:", args.text, flush=True)
        for k, v in stats.items():
            print(f"  {k}: {v}", flush=True)
        print("wrote", args.out, "samples", len(audio), flush=True)
        print("OK", flush=True)
    finally:
        tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
