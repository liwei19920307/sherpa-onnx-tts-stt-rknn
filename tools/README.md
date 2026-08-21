# Offline tools (run on a machine with rknn-toolkit2 installed)

| Script | Purpose |
|--------|---------|
| `compile_kokoro_rk3566.py` | `kokoro_decoder.onnx` → `decoder_rk3566.rknn` |
| `compile_melo_rk3566.py` | Melo `decoder.onnx` → `decoder_rk3566.rknn` |

Set `ONNX` / `OUT` (and optionally `T_FIX` for Kokoro). These scripts do **not** ship Rockchip's proprietary toolkit.
