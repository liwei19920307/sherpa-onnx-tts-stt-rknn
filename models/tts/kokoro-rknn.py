"""Builtin Kokoro hybrid TTS: encoder/har ONNX (CPU) + decoder RKNN (NPU)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from kokoro_rknn_tts import KokoroRknnTts

_LOGGER = logging.getLogger("sherpa_onnx_kokoro_rknn_tts")


class GeneratedAudio:
    __slots__ = ("samples", "sample_rate")

    def __init__(self, samples: np.ndarray, sample_rate: int):
        self.samples = samples
        self.sample_rate = sample_rate


def _list_voices(model_dir: Path, preferred: str = "zf_001") -> list[str]:
    names = []
    for folder in ("voices", "voices_npy"):
        names.extend(p.stem for p in (model_dir / folder).glob("*.npy"))
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    def key(n: str) -> tuple:
        if n == preferred:
            return (0, n)
        if n.startswith("zf_"):
            return (1, n)
        if n.startswith("zm_"):
            return (2, n)
        return (3, n)

    return sorted(unique, key=key)


class KokoroRknnOfflineTts:
    def __init__(self, model_dir: Path, voice: str = "zf_001", speed: float = 1.0):
        self._voices = _list_voices(model_dir, preferred=voice)
        if voice not in self._voices and self._voices:
            voice = self._voices[0]
        self._tts = KokoroRknnTts(model_dir, voice=voice, speed=speed)
        self.sample_rate = self._tts.sample_rate

    def generate(self, text: str, sid: int = 0, speed: float = 1.0) -> GeneratedAudio:
        if self._voices:
            voice = self._voices[int(sid) % len(self._voices)]
        else:
            voice = self._tts.voice
        old = self._tts.speed
        try:
            if speed and speed > 0:
                self._tts.speed = float(speed)
            audio, stats = self._tts.synthesize(text, voice=voice)
            _LOGGER.info(
                "kokoro-rknn voice=%s rtf=%s decoder_ms=%s npu=%s",
                voice,
                stats.get("rtf"),
                stats.get("decoder_ms"),
                stats.get("npu"),
            )
        finally:
            self._tts.speed = old
        return GeneratedAudio(samples=audio.astype(np.float32), sample_rate=self.sample_rate)


def _resolve_model_dir(cli_args) -> Path:
    explicit = os.environ.get("KOKORO_RKNN_DIR") or getattr(cli_args, "kokoro_rknn_dir", None)
    if explicit:
        return Path(explicit)
    tts_model_dir = os.environ.get("TTS_MODEL_DIR", "/tts-models")
    candidates = [
        Path(tts_model_dir) / "kokoro-rknn",
        Path(tts_model_dir) / "kokoro-rknn-zh",
    ]
    for c in candidates:
        has_enc = (c / "encoder.onnx").is_file() or (c / "kokoro_encoder.onnx").is_file()
        has_dec = any(
            (c / n).is_file()
            for n in (
                "decoder_rk3588.rknn",
                "decoder_rk3566.rknn",
                "kokoro_decoder.rknn",
                "decoder.rknn",
            )
        )
        if has_enc and has_dec:
            return c
    raise FileNotFoundError(
        "Kokoro RKNN model dir not found. Set KOKORO_RKNN_DIR or place "
        "encoder.onnx + har_generator.onnx + decoder_rk3588.rknn "
        "(and/or decoder_rk3566.rknn) under TTS_MODEL_DIR/kokoro-rknn"
    )


def load(cli_args):
    model_dir = _resolve_model_dir(cli_args)
    voice = os.environ.get("KOKORO_VOICE") or getattr(cli_args, "kokoro_voice", None) or "zf_001"
    speed = float(getattr(cli_args, "speed", 1.0) or 1.0)
    _LOGGER.info("Loading Kokoro RKNN TTS from %s voice=%s", model_dir, voice)
    return KokoroRknnOfflineTts(model_dir, voice=voice, speed=speed)
