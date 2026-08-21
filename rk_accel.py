"""Rockchip accelerator detection for sherpa-onnx.

What helps this TTS/STT addon on RK3566 / RK3588:

  NPU (RKNPU / RKNN)  -> ASR (STT) with .rknn models via --provider=rknn
  CPU                 -> TTS (Kokoro/Matcha/VITS) — no official .rknn TTS path here
  Mali GPU            -> not used by sherpa-onnx (no OpenCL/CUDA-on-Mali provider)
  MPP / VPU / RGA     -> video encode/decode/scale only — unused by audio STT/TTS
  ffmpeg / pydub      -> audio container decode stays on CPU

For RK3588 NPU core selection, sherpa reinterprets num_threads when provider=rknn:
  1  -> AUTO, 0 -> core0, -1 -> core1, -2 -> core2, -3 -> 0+1, -4 -> 0+1+2
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_LOGGER = logging.getLogger("sherpa_onnx_rk_accel")

_RK_SOCS = ("rk3588", "rk3576", "rk3568", "rk3566", "rk3562")


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def detect_rk_soc() -> str | None:
    """Return e.g. 'rk3588' / 'rk3566' or None."""
    forced = (os.environ.get("RK_SOC") or os.environ.get("RK_PLATFORM") or "").strip().lower()
    if forced:
        forced = forced.replace("rockchip,", "").replace(",", "")
        for soc in _RK_SOCS:
            if soc in forced:
                return soc

    compatible = read_text("/proc/device-tree/compatible").lower().replace("\x00", " ")
    if not compatible.strip():
        compatible = read_text("/sys/firmware/devicetree/base/compatible").lower().replace(
            "\x00", " "
        )
    for soc in _RK_SOCS:
        if soc in compatible:
            return soc

    cpuinfo = read_text("/proc/cpuinfo").lower()
    for soc in _RK_SOCS:
        if soc in cpuinfo:
            return soc
    return None


def npu_device_present() -> bool:
    candidates = (
        "/dev/rknpu",
        "/dev/rknpu0",
        "/sys/class/misc/rknpu",
        "/sys/kernel/debug/rknpu/load",
        "/sys/kernel/debug/rknpu/version",
    )
    return any(Path(p).exists() for p in candidates)


def librknnrt_present() -> bool:
    for p in (
        "/usr/lib/librknnrt.so",
        "/lib/librknnrt.so",
        "/usr/lib/aarch64-linux-gnu/librknnrt.so",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ):
        if not p:
            continue
        if ":" in p:
            for part in p.split(":"):
                if part and Path(part, "librknnrt.so").exists():
                    return True
        elif Path(p).is_file() or Path(p, "librknnrt.so").exists():
            return True
    return False


def rknn_ready() -> bool:
    return npu_device_present() or librknnrt_present() or detect_rk_soc() is not None


def resolve_provider(explicit: str | None = None) -> str:
    """Pick execution provider: rknn | cuda | cpu.

    Priority:
      1) explicit CLI/env PROVIDER (unless auto)
      2) NVIDIA GPU
      3) Rockchip NPU
      4) CPU
    """
    explicit = (explicit or os.environ.get("PROVIDER") or "auto").strip().lower()
    if explicit in {"rknn", "cuda", "cpu"}:
        return explicit

    # nvidia (same behavior as upstream addon)
    if Path("/usr/bin/nvidia-smi").exists() or Path("/dev/nvidia0").exists():
        try:
            import subprocess

            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            _LOGGER.info("NVIDIA GPU detected -> provider=cuda")
            return "cuda"
        except Exception:
            pass

    soc = detect_rk_soc()
    if soc and rknn_ready():
        _LOGGER.info("Rockchip %s NPU detected -> provider=rknn", soc)
        return "rknn"
    if soc:
        _LOGGER.warning(
            "Rockchip %s detected but NPU runtime/device not confirmed; "
            "set PROVIDER=rknn and mount /dev/rknpu + librknnrt.so if needed",
            soc,
        )
        # still prefer rknn on Rockchip aarch64 boards when user asked for auto
        if os.environ.get("FORCE_RKNN", "").lower() in {"1", "true", "yes"}:
            return "rknn"

    _LOGGER.info("Using provider=cpu")
    return "cpu"


def default_rknn_stt_model(soc: str | None = None, kind: str = "sense-voice") -> str | None:
    """Suggest a downloadable official .rknn ASR package name."""
    soc = soc or detect_rk_soc()
    if not soc:
        return None
    # Official releases use rk3566 for the 356x family (3566/3568).
    if soc in ("rk3568", "rk3562"):
        soc = "rk3566"
    if kind == "paraformer":
        # 15s is a good HA Assist utterance length default
        return f"sherpa-onnx-{soc}-15-seconds-paraformer-zh-2025-10-07"
    # SenseVoice multilingual — prefer 5s (HA Assist utterances); override via full folder name
    return f"sherpa-onnx-{soc}-5-seconds-sense-voice-zh-en-ja-ko-yue-2025-09-09"


def is_rknn_model_name(name: str | None) -> bool:
    if not name:
        return False
    return bool(re.search(r"sherpa-onnx-rk\d{4}", name)) and (
        "sense-voice" in name or "paraformer" in name or "zipformer" in name
    )


def model_kind(name: str) -> str | None:
    if "sense-voice" in name:
        return "sense-voice"
    if "paraformer" in name:
        return "paraformer"
    if "zipformer" in name:
        return "zipformer"
    return None


def resolve_stt_model_for_provider(stt_model: str | None, provider: str) -> str | None:
    """If provider is rknn and model is a CPU ONNX default, swap to a chip-matched RKNN pack."""
    if provider != "rknn":
        return stt_model
    if stt_model and is_rknn_model_name(stt_model):
        return stt_model

    # Map built-in CPU paraformer names to RKNN paraformer
    if not stt_model or "paraformer" in (stt_model or ""):
        mapped = default_rknn_stt_model(kind="paraformer")
        if mapped:
            _LOGGER.info("Auto-select RKNN STT model: %s (was %s)", mapped, stt_model)
            return mapped

    mapped = default_rknn_stt_model(kind="sense-voice")
    if mapped:
        _LOGGER.info("Auto-select RKNN STT model: %s (was %s)", mapped, stt_model)
        return mapped
    return stt_model
