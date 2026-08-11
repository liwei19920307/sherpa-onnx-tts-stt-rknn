# sherpa-onnx-tts-stt/model_utils.py
import os
import subprocess
import logging
import sherpa_onnx
import importlib
import sys

_LOGGER = logging.getLogger("sherpa_onnx_model_utils")


def _download_model(model_url, model_dir, model):
    """Downloads and extracts the model."""
    if not os.path.exists(os.path.join(model_dir, model)):
        _LOGGER.info(f"Downloading model: {model_url}")
        os.makedirs(os.path.join(model_dir, model), exist_ok=True)

        # Use curl (or wget) for download and extraction (more robust than Python libraries for large files)
        ext = ".tar.bz2" if model_url.endswith(".tar.bz2") else ".tar.gz"
        archive_path = os.path.join(model_dir, model, f"{model}{ext}")
        try:
            subprocess.check_call(
                [
                    "curl",
                    "-L",
                    model_url,
                    "-o",
                    archive_path,
                ]
            )
            _LOGGER.info(f"Downloaded model: {model_url}, Extracting...")
            subprocess.check_call(
                [
                    "tar",
                    "-xvf",
                    archive_path,
                    "-C",
                    model_dir,
                ]
            )
            os.remove(archive_path)  # Clean up
            _LOGGER.info(f"Download and extract Done. Cleaned up.")
        except subprocess.CalledProcessError as e:
            _LOGGER.error(f"Error downloading or extracting model: {e}")
            raise  #  Re-raise to stop add-on startup on failure
    else:
        _LOGGER.info(f"{model} model already exists.")


def fetch_stt_model(stt_model_dir, model):
    # --- STT Model ---
    stt_model_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model}.tar.bz2"
    _download_model(stt_model_url, stt_model_dir, model)


def fetch_tts_model(tts_model_dir, model):
    # --- TTS Model ---
    tts_model_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{model}.tar.bz2"
    _download_model(tts_model_url, tts_model_dir, model)

def fetch_vocoder_model(model_dir, model):
    # --- Vocoder Model ---
    model_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/{model}"
    if not os.path.exists(os.path.join(model_dir, model)):
        _LOGGER.info("Downloading model: %s", model_url)
        os.makedirs(model_dir, exist_ok=True)

        # Use curl (or wget) for download and extraction (more robust than Python libraries for large files)
        try:
            subprocess.check_call(
                [
                    "curl",
                    "-L",
                    model_url,
                    "-o",
                    os.path.join(model_dir, model),
                ]
            )
            _LOGGER.info("Downloaded model: %s", model_url)
        except subprocess.CalledProcessError as e:
            _LOGGER.error("Error downloading model: %s", e)
            raise  #  Re-raise to stop add-on startup on failure
    else:
        _LOGGER.info("%s model already exists.", model)

def load_module(file):
    spec = importlib.util.spec_from_file_location("model", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_builtin_model(model, language, project_dir, model_type):
    if model:
        candidate = os.path.join(
            project_dir,
            "models",
            model_type,
            f"{model}.py",
        )
        if os.path.exists(candidate):
            return candidate
        # Rockchip packages: sherpa-onnx-rk3588-*-sense-voice-* etc.
        # Map package names to shared loaders without one file per release.
        if model_type == "stt":
            from rk_accel import is_rknn_model_name, model_kind

            if is_rknn_model_name(model):
                kind = model_kind(model)
                if kind == "sense-voice":
                    return os.path.join(project_dir, "models", "stt", "rknn-sense-voice.py")
                if kind == "paraformer":
                    return os.path.join(
                        project_dir, "models", "stt", "rknn-paraformer-zh.py"
                    )
                if kind == "zipformer":
                    # reuse sense-voice entry point routing via rknn_stt
                    return os.path.join(project_dir, "models", "stt", "rknn-sense-voice.py")
        return candidate
    elif language:
        model = os.path.join(project_dir, "models", model_type, "lang", language)
        if os.path.exists(model):
            return os.path.realpath(model)
    return None


def initialize_models(cli_args):
    """Initializes STT and TTS models based on CLI arguments."""

    stt_model_dir = "/stt-models"
    tts_model_dir = "/tts-models"
    # Allow host/bind-mount overrides for local development
    if os.environ.get("STT_MODEL_DIR"):
        stt_model_dir = os.environ["STT_MODEL_DIR"]
    if os.environ.get("TTS_MODEL_DIR"):
        tts_model_dir = os.environ["TTS_MODEL_DIR"]
    os.environ.setdefault("STT_MODEL_DIR", stt_model_dir)
    os.environ.setdefault("TTS_MODEL_DIR", tts_model_dir)
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # Matcha/Kokoro stay ONNX on CPU when provider=rknn.
    # Hybrid Melo/Piper use NPU for the decoder via librknnrt.
    tts_model_name = getattr(cli_args, "tts_model", "") or ""
    hybrid_rknn = tts_model_name in {
        "piper-rknn-zh",
        "piper_rknn_zh",
        "piper-zh-rknn",
        "melo-rknn-zh_en",
        "melo_rknn_zh_en",
        "melo-rknn",
    }
    tts_provider = "cpu" if getattr(cli_args, "provider", None) == "rknn" else cli_args.provider
    if getattr(cli_args, "provider", None) == "rknn":
        if hybrid_rknn:
            _LOGGER.info(
                "provider=rknn: STT on NPU; TTS=%s uses NPU decoder + CPU encoder",
                tts_model_name,
            )
        else:
            _LOGGER.info(
                "provider=rknn: STT can use NPU; Matcha/Kokoro/Melo-CPU stay on CPU "
                "(set TTS_MODEL=melo-rknn-zh_en or piper-rknn-zh for NPU decoder)"
            )

    # STT Initialization (adjust paths as needed for extracted model)
    try:
        if cli_args.custom_stt_model_eval != "null":
            if cli_args.stt_model:
                fetch_stt_model(stt_model_dir, cli_args.stt_model)
            stt_model = eval(cli_args.custom_stt_model_eval)
        else:
            model_file = find_builtin_model(
                cli_args.stt_model, cli_args.language, project_dir, "stt"
            )
            if model_file and os.path.exists(model_file):
                stt_model = load_module(model_file).load(cli_args)
            else:
                # Fallback: treat unknown sherpa-onnx-rk* names via rknn_stt
                from rk_accel import is_rknn_model_name
                from rknn_stt import load_rknn_stt

                if is_rknn_model_name(cli_args.stt_model):
                    stt_model = load_rknn_stt(cli_args)
                else:
                    stt_model = None
    except Exception as e:
        _LOGGER.critical("Failed to initialize custom STT model: %s", e)
        raise

    try:
        # TTS Initialization (hybrid NPU loaders ignore sherpa provider)
        stt_provider = getattr(cli_args, "provider", "cpu")
        cli_args.provider = tts_provider
        if cli_args.custom_tts_model_eval != "null":
            if cli_args.tts_model:
                fetch_tts_model(tts_model_dir, cli_args.tts_model)
            tts_model = eval(cli_args.custom_tts_model_eval)
        else:
            model_file = find_builtin_model(
                cli_args.tts_model, cli_args.language, project_dir, "tts"
            )
            tts_model = load_module(model_file).load(cli_args) if model_file else None
        cli_args.provider = stt_provider
    except Exception as e:
        cli_args.provider = getattr(cli_args, "provider", "cpu")
        _LOGGER.critical("Failed to initialize custom TTS model: %s", e)
        raise

    if not (tts_model or stt_model):
        _LOGGER.critical("No models loaded")
        raise Exception("No models loaded")

    return stt_model, tts_model
