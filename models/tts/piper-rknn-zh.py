"""Builtin Piper-zh hybrid TTS: ONNX encoder (CPU) + RKNN decoder (NPU)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from piper_rknn_tts import PiperRknnTts

_LOGGER = logging.getLogger("sherpa_onnx_piper_rknn_tts")


class GeneratedAudio:
    __slots__ = ("samples", "sample_rate")

    def __init__(self, samples: np.ndarray, sample_rate: int):
        self.samples = samples
        self.sample_rate = sample_rate


class PiperRknnOfflineTts:
    """Duck-typed OfflineTts compatible with run.py / api.py generate() calls."""

    def __init__(self, model_dir: Path):
        self._tts = PiperRknnTts(model_dir)
        self.sample_rate = self._tts.sample_rate

    def generate(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
    ) -> GeneratedAudio:
        # Piper length_scale: larger => slower speech. Map speed>1 to shorter.
        old = self._tts.length_scale
        try:
            if speed and speed > 0:
                self._tts.length_scale = old / float(speed)
            audio, stats = self._tts.synthesize(text)
            _LOGGER.info(
                "piper-rknn synthesize rtf=%s decoder_ms=%s npu=%s",
                stats.get("rtf"),
                stats.get("decoder_ms"),
                stats.get("npu"),
            )
        finally:
            self._tts.length_scale = old
        return GeneratedAudio(samples=audio.astype(np.float32), sample_rate=self.sample_rate)


def _resolve_model_dir(cli_args) -> Path:
    explicit = os.environ.get("PIPER_RKNN_DIR") or getattr(cli_args, "piper_rknn_dir", None)
    if explicit:
        return Path(explicit)
    tts_model_dir = os.environ.get("TTS_MODEL_DIR", "/tts-models")
    candidates = [
        Path(tts_model_dir) / "piper-zh",
        Path(tts_model_dir) / "piper-rknn-zh",
        Path("/root/models/piper-zh"),
    ]
    for c in candidates:
        has_enc = (c / "encoder.onnx").is_file()
        has_dec = (c / "decoder_rk3566.rknn").is_file() or (c / "decoder_rk3588.rknn").is_file()
        if has_enc and has_dec:
            return c
    raise FileNotFoundError(
        "Piper RKNN model dir not found. Set PIPER_RKNN_DIR or place "
        "encoder.onnx + decoder_rk*.rknn + config.json under TTS_MODEL_DIR/piper-zh"
    )


def load(cli_args):
    model_dir = _resolve_model_dir(cli_args)
    _LOGGER.info("Loading Piper RKNN TTS from %s", model_dir)
    return PiperRknnOfflineTts(model_dir)
