"""TTS_MODEL=kokoro-rknn-zh alias."""
import importlib.util
import os

_here = os.path.dirname(os.path.realpath(__file__))
_spec = importlib.util.spec_from_file_location(
    "kokoro_rknn_canonical", os.path.join(_here, "kokoro-rknn.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load = _mod.load
