#!/usr/bin/env python3
"""Compile Kokoro decoder ONNX → RK3566 RKNN (fp16). Requires rknn-toolkit2.

Example:
  ONNX=./kokoro_decoder.onnx OUT=./decoder_rk3566.rknn python tools/compile_kokoro_rk3566.py
"""
import os
from pathlib import Path
from rknn.api import RKNN

onnx = Path(os.environ.get("ONNX", "kokoro_decoder.onnx"))
out = Path(os.environ.get("OUT", "decoder_rk3566.rknn"))
t_fix = int(os.environ.get("T_FIX", "50"))
har_F = 2 * t_fix * 300 // 5 + 1
print("input", onnx, "MB", round(onnx.stat().st_size / 1e6, 1))
r = RKNN(verbose=False)
r.config(target_platform="rk3566")
ret = r.load_onnx(
    model=str(onnx),
    inputs=["asr", "F0", "N", "s", "har"],
    input_size_list=[
        [1, 512, t_fix],
        [1, 2 * t_fix],
        [1, 2 * t_fix],
        [1, 128],
        [1, 22, har_F],
    ],
)
print("load_onnx", ret)
if ret != 0:
    raise SystemExit(ret)
if r.build(do_quantization=False) != 0:
    raise SystemExit("build failed")
if r.export_rknn(str(out)) != 0:
    raise SystemExit("export failed")
r.release()
print("OK wrote", out, "MB", round(out.stat().st_size / 1e6, 1))
