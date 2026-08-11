"""Builtin entry for auto chip-matched SenseVoice RKNN STT."""

from rk_accel import default_rknn_stt_model
from rknn_stt import load_rknn_stt


def load(cli_args):
    model = cli_args.stt_model
    if not model or model in {
        "rknn-sense-voice",
        "rknn_sense_voice",
        "auto-rknn-sense-voice",
    }:
        model = default_rknn_stt_model(kind="sense-voice")
        cli_args.stt_model = model
    return load_rknn_stt(cli_args, model)
