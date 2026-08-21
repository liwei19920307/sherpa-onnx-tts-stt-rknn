#!/usr/bin/env python3
"""Compile Melo decoder.onnx → RK3566 RKNN (fp16). Requires rknn-toolkit2.

Example:
  ONNX=./decoder.onnx OUT=./decoder_rk3566.rknn python tools/compile_melo_rk3566.py
"""
import os
from pathlib import Path
from rknn.api import RKNN

onnx = Path(os.environ.get("ONNX", "decoder.onnx"))
out = Path(os.environ.get("OUT", "decoder_rk3566.rknn"))
print("input", onnx, "MB", round(onnx.stat().st_size / 1e6, 1))
r = RKNN(verbose=False)
r.config(target_platform="rk3566", float_dtype="float16", optimization_level=3)
ret = r.load_onnx(model=str(onnx))
print("load_onnx", ret)
if ret != 0:
    raise SystemExit(ret)
if r.build(do_quantization=False) != 0:
    raise SystemExit("build failed")
if r.export_rknn(str(out)) != 0:
    raise SystemExit("export failed")
r.release()
print("OK wrote", out, "MB", round(out.stat().st_size / 1e6, 1))
