#!/usr/bin/env python3
"""Minimal RKNNLite-compatible wrapper via ctypes + librknnrt.so (py3.14 friendly)."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Sequence

import numpy as np


class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RKNNLite:
    """Drop-in subset of rknnlite.api.RKNNLite used by melotts_rknn.py."""

    RKNN_TENSOR_FLOAT32 = 0
    RKNN_TENSOR_NCHW = 0
    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1
    RKNN_QUERY_OUTPUT_ATTR = 2
    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1 = 3
    NPU_CORE_0_1_2 = 7
    NPU_CORE_ALL = 65535

    def __init__(self, lib_path: str | None = None):
        self.ctx = ctypes.c_void_p()
        self._model_bytes: bytes | None = None
        self.lib = ctypes.CDLL(lib_path or "librknnrt.so")
        self.lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.lib.rknn_init.restype = ctypes.c_int
        self.lib.rknn_query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.lib.rknn_query.restype = ctypes.c_int
        self.lib.rknn_inputs_set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RKNNInput),
        ]
        self.lib.rknn_inputs_set.restype = ctypes.c_int
        self.lib.rknn_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.rknn_run.restype = ctypes.c_int
        self.lib.rknn_outputs_get.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RKNNOutput),
            ctypes.c_void_p,
        ]
        self.lib.rknn_outputs_get.restype = ctypes.c_int
        self.lib.rknn_outputs_release.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RKNNOutput),
        ]
        self.lib.rknn_outputs_release.restype = ctypes.c_int
        self.lib.rknn_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rknn_destroy.restype = ctypes.c_int
        # Optional multi-core (RK3588)
        if hasattr(self.lib, "rknn_set_core_mask"):
            self.lib.rknn_set_core_mask.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.rknn_set_core_mask.restype = ctypes.c_int
        self.n_input = 0
        self.n_output = 0
        self._in_attrs: List[RKNNTensorAttr] = []
        self._out_attrs: List[RKNNTensorAttr] = []
        self._model_buf = None

    def load_rknn(self, model_path: str) -> int:
        self._model_bytes = Path(model_path).read_bytes()
        return 0

    def init_runtime(self, core_mask: int = 0, **_kwargs) -> int:
        if not self._model_bytes:
            return -1
        # Keep buffer alive for rknn_init lifetime
        self._model_buf = ctypes.create_string_buffer(self._model_bytes)
        ret = self.lib.rknn_init(
            ctypes.byref(self.ctx), self._model_buf, len(self._model_bytes), 0, None
        )
        if ret != 0:
            return ret
        if core_mask and hasattr(self.lib, "rknn_set_core_mask"):
            # Best-effort; ignore failure on single-core SoCs
            self.lib.rknn_set_core_mask(self.ctx, int(core_mask))
        io = RKNNInputOutputNum()
        ret = self.lib.rknn_query(
            self.ctx,
            self.RKNN_QUERY_IN_OUT_NUM,
            ctypes.byref(io),
            ctypes.sizeof(io),
        )
        if ret != 0:
            return ret
        self.n_input = int(io.n_input)
        self.n_output = int(io.n_output)
        self._in_attrs = []
        for i in range(self.n_input):
            attr = RKNNTensorAttr()
            attr.index = i
            ret = self.lib.rknn_query(
                self.ctx,
                self.RKNN_QUERY_INPUT_ATTR,
                ctypes.byref(attr),
                ctypes.sizeof(attr),
            )
            if ret != 0:
                return ret
            self._in_attrs.append(attr)
        self._out_attrs = []
        for i in range(self.n_output):
            attr = RKNNTensorAttr()
            attr.index = i
            ret = self.lib.rknn_query(
                self.ctx,
                self.RKNN_QUERY_OUTPUT_ATTR,
                ctypes.byref(attr),
                ctypes.sizeof(attr),
            )
            if ret != 0:
                return ret
            self._out_attrs.append(attr)
        return 0

    def inference(self, inputs: Sequence[np.ndarray], data_format=None):
        del data_format
        assert len(inputs) == self.n_input
        c_inputs = (RKNNInput * self.n_input)()
        keep_alive = []
        for i, arr in enumerate(inputs):
            x = np.ascontiguousarray(arr, dtype=np.float32)
            keep_alive.append(x)
            c_inputs[i].index = i
            c_inputs[i].buf = x.ctypes.data_as(ctypes.c_void_p)
            c_inputs[i].size = x.nbytes
            c_inputs[i].pass_through = 0
            c_inputs[i].type = self.RKNN_TENSOR_FLOAT32
            c_inputs[i].fmt = self.RKNN_TENSOR_NCHW
        ret = self.lib.rknn_inputs_set(self.ctx, self.n_input, c_inputs)
        if ret != 0:
            raise RuntimeError(f"rknn_inputs_set failed: {ret}")
        ret = self.lib.rknn_run(self.ctx, None)
        if ret != 0:
            raise RuntimeError(f"rknn_run failed: {ret}")
        outs = (RKNNOutput * self.n_output)()
        for i in range(self.n_output):
            outs[i].want_float = 1
            outs[i].is_prealloc = 0
        ret = self.lib.rknn_outputs_get(self.ctx, self.n_output, outs, None)
        if ret != 0:
            raise RuntimeError(f"rknn_outputs_get failed: {ret}")
        results = []
        for i in range(self.n_output):
            size = outs[i].size
            buf = ctypes.string_at(outs[i].buf, size)
            results.append(np.frombuffer(buf, dtype=np.float32).copy())
        self.lib.rknn_outputs_release(self.ctx, self.n_output, outs)
        return results

    def release(self) -> None:
        if self.ctx:
            self.lib.rknn_destroy(self.ctx)
            self.ctx = ctypes.c_void_p()
