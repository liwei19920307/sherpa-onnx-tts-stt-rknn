"""Shared loaders for Rockchip .rknn ASR packages from k2-fsa/sherpa-onnx."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import sherpa_onnx

import model_utils
from rk_accel import model_kind

_LOGGER = logging.getLogger("sherpa_onnx_rknn_stt")

# re-export for callers that imported from the old path
__all__ = ["load_rknn_stt"]


def _find_model_file(model_dir: Path, stems: tuple[str, ...]) -> str:
    for stem in stems:
        for name in (f"{stem}.rknn", f"{stem}.onnx", f"{stem}.int8.onnx"):
            candidate = model_dir / name
            if candidate.is_file():
                return str(candidate)
    # fall back to first .rknn
    matches = sorted(model_dir.glob("*.rknn"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No .rknn model found under {model_dir}")


def load_rknn_stt(cli_args, stt_model: str | None = None):
    stt_model = stt_model or cli_args.stt_model
    if not stt_model:
        raise ValueError("stt_model is required for RKNN loader")

    stt_model_dir = os.environ.get("STT_MODEL_DIR", "/stt-models")
    model_utils.fetch_stt_model(stt_model_dir, stt_model)
    package_dir = Path(stt_model_dir) / stt_model
    tokens = str(package_dir / "tokens.txt")
    kind = model_kind(stt_model)
    provider = getattr(cli_args, "provider", "rknn") or "rknn"
    num_threads = int(getattr(cli_args, "stt_thread_num", 1) or 1)
    debug = bool(getattr(cli_args, "debug", False))
    rule_fsts = (
        os.path.join("/app/", "itn_zh_number.fst")
        if getattr(cli_args, "stt_builtin_auto_convert_number", False)
        else ""
    )

    if provider != "rknn":
        _LOGGER.warning(
            "Loading RKNN package %s with provider=%s (expected rknn)",
            stt_model,
            provider,
        )

    if kind == "sense-voice":
        model = _find_model_file(package_dir, ("model",))
        _LOGGER.info("SenseVoice RKNN model=%s provider=%s", model, provider)
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
            use_itn=True,
            debug=debug,
            rule_fsts=rule_fsts,
        )

    if kind == "paraformer":
        model = _find_model_file(package_dir, ("model", "model.int8", "encoder"))
        _LOGGER.info("Paraformer RKNN model=%s provider=%s", model, provider)
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model,
            tokens=tokens,
            decoding_method="greedy_search",
            provider=provider,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            debug=debug,
            rule_fsts=rule_fsts,
        )

    if kind == "zipformer":
        encoder = _find_model_file(package_dir, ("encoder",))
        decoder = str(package_dir / "decoder.rknn")
        joiner = str(package_dir / "joiner.rknn")
        _LOGGER.info(
            "Streaming Zipformer RKNN encoder=%s provider=%s", encoder, provider
        )
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=num_threads,
            provider=provider,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            debug=debug,
        )

    raise ValueError(f"Unsupported RKNN STT model kind for: {stt_model}")
