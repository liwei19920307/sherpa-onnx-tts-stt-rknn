# Third-party notices

This repository is a fork of [ptbsare/sherpa-onnx-tts-stt](https://github.com/ptbsare/sherpa-onnx-tts-stt)
(Home Assistant / Wyoming sherpa-onnx TTS+STT addon). The upstream repo did not
declare a SPDX license at the time of forking; treat upstream code accordingly
and retain attribution.

## Critical: AGPL-3.0 code in-tree

| Component | Source | License | Notes |
|-----------|--------|---------|-------|
| `melo_rknn/` (inference helpers derived from MeloTTS-RKNN2) | [happyme531/MeloTTS-RKNN2](https://huggingface.co/happyme531/MeloTTS-RKNN2) | **AGPL-3.0** | Shipping this tree (or a Docker image that includes it) triggers AGPL obligations. This project is therefore released under **AGPL-3.0**. |

If you cannot accept AGPL, remove Melo RKNN support (`melo_rknn/`, `melo_rknn_tts.py`,
`models/tts/melo-rknn-zh_en.py`) and use only Kokoro / CPU Melo ONNX paths after
your own license review.

## Models (not stored in this git repo)

Weights / `.rknn` / `.onnx` packs are **not** committed. Download separately; each
has its own terms:

| Asset | Typical source | License (as published) |
|-------|----------------|------------------------|
| Kokoro-82M / v1.1-zh | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) | Apache-2.0 |
| MeloTTS (original) | [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) | MIT |
| Melo RKNN decoder ONNX/RKNN | happyme531/MeloTTS-RKNN2 | AGPL-3.0 (repo + conversion pipeline) |
| SenseVoice / Paraformer RKNN ASR | [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) | Apache-2.0 (sherpa-onnx); check FunAudioLLM/SenseVoice (MIT) for model lineage |
| Piper (optional hybrid) | Piper / community conversions | Check each voice pack |

## Runtime / SDK (not open-source)

| Component | Vendor | Notes |
|-----------|--------|-------|
| `librknnrt.so` / RKNPU runtime | Rockchip | Proprietary. Mount host library at runtime; do not assume redistributable without Rockchip terms. |
| rknn-toolkit2 | Rockchip | Proprietary. Only needed to *convert* models; not required to run this app. |
| `sherpa_onnx-*-aarch64.whl` (RKNN build) | k2-fsa | Apache-2.0; fetch from official docs, do not commit `.whl` |

## Other open-source dependencies (runtime)

Examples (not exhaustive): `wyoming`, `fastapi`, `onnxruntime`, `misaki`, `espeak-ng`,
`numpy`, `jieba`, `pypinyin` — each under their own OSS licenses (typically Apache/MIT/BSD).

## Kokoro hybrid pipeline

`kokoro_rknn_tts.py` follows the hybrid encoder(CPU)+decoder(NPU) approach used in
community Kokoro-on-Rockchip projects (e.g. marty1885/kokoro-server, which had **no
declared license** when reviewed). Weights remain Apache-2.0 Kokoro; treat any
copied conversion logic carefully and prefer reimplementation attribution.

## Docker Hub images

Prebuilt images that **bake** Melo RKNN code + AGPL-origin decoder artefacts inherit
AGPL distribution duties (source offer corresponding to the image). Prefer linking
this git repo as the Corresponding Source.
