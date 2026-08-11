"""Builtin entry for auto chip-matched Paraformer-zh RKNN STT."""

from rk_accel import default_rknn_stt_model
from rknn_stt import load_rknn_stt


def load(cli_args):
    model = cli_args.stt_model
    if not model or model in {
        "rknn-paraformer-zh",
        "rknn_paraformer_zh",
        "auto-rknn-paraformer-zh",
    }:
        model = default_rknn_stt_model(kind="paraformer")
        cli_args.stt_model = model
    return load_rknn_stt(cli_args, model)
