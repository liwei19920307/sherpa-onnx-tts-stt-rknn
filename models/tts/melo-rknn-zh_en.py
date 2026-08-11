"""Builtin Melo hybrid TTS: ONNX encoder (CPU) + RKNN decoder (NPU)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from melo_rknn_tts import MeloRknnTts

_LOGGER = logging.getLogger("sherpa_onnx_melo_rknn_tts")


class GeneratedAudio:
    __slots__ = ("samples", "sample_rate")

    def __init__(self, samples: np.ndarray, sample_rate: int):
        self.samples = samples
        self.sample_rate = sample_rate


class MeloRknnOfflineTts:
    """Duck-typed OfflineTts compatible with run.py / api.py generate() calls."""

    def __init__(self, model_dir: Path, speed: float = 0.8):
        self._base_speed = float(speed)
        self._tts = MeloRknnTts(model_dir, speed=self._base_speed)
        self.sample_rate = self._tts.sample_rate

    def generate(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
    ) -> GeneratedAudio:
        del sid
        old = self._tts.speed
        try:
            # Addon speed: 1.0 = model default; >1 faster => smaller melo speed scale.
            if speed and speed > 0:
                self._tts.speed = self._base_speed * float(speed)
            audio, stats = self._tts.synthesize(text)
            _LOGGER.info(
                "melo-rknn synthesize rtf=%s decoder_ms=%s npu=%s core_mask=%s",
                stats.get("rtf"),
                stats.get("decoder_ms"),
                stats.get("npu"),
                stats.get("core_mask"),
            )
        finally:
            self._tts.speed = old
        return GeneratedAudio(samples=audio.astype(np.float32), sample_rate=self.sample_rate)


def _resolve_model_dir(cli_args) -> Path:
    explicit = os.environ.get("MELO_RKNN_DIR") or getattr(cli_args, "melo_rknn_dir", None)
    if explicit:
        return Path(explicit)
    tts_model_dir = os.environ.get("TTS_MODEL_DIR", "/tts-models")
    candidates = [
        Path(tts_model_dir) / "melo-rknn",
        Path(tts_model_dir) / "melo-rknn-zh_en",
        Path("/root/melo-rknn"),
    ]
    for c in candidates:
        if (c / "encoder.onnx").is_file() and (
            (c / "decoder_rk3588.rknn").is_file() or (c / "decoder_rk3566.rknn").is_file()
        ):
            return c
    raise FileNotFoundError(
        "Melo RKNN model dir not found. Set MELO_RKNN_DIR or place "
        "encoder.onnx + decoder_rk*.rknn + g.bin + lexicon.txt + tokens.txt under "
        "TTS_MODEL_DIR/melo-rknn"
    )


def load(cli_args):
    model_dir = _resolve_model_dir(cli_args)
    speed = float(getattr(cli_args, "speed", 0.8) or 0.8)
    # Map HA speed (typically 1.0) to Melo length scale baseline 0.8 when speed~1
    melo_speed = 0.8 if abs(speed - 1.0) < 1e-6 else max(0.3, min(2.0, 0.8 * speed))
    _LOGGER.info("Loading Melo RKNN TTS from %s (speed=%s)", model_dir, melo_speed)
    return MeloRknnOfflineTts(model_dir, speed=melo_speed)
