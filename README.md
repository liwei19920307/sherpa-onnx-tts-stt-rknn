# sherpa-onnx-tts-stt (Rockchip RKNN fork)

Wyoming + OpenAI-compatible **offline TTS/STT** for Home Assistant, with **Rockchip NPU**
acceleration on RK3566 / RK3568 / RK3588.

Fork of [ptbsare/sherpa-onnx-tts-stt](https://github.com/ptbsare/sherpa-onnx-tts-stt).

**License: [AGPL-3.0](./LICENSE)** — see [NOTICE.md](./NOTICE.md) (Melo RKNN path is AGPL-derived).

## Features

- **STT (NPU):** SenseVoice / Paraformer official sherpa-onnx `.rknn` packs (auto SoC pick)
- **TTS (NPU decoder):**
  - `kokoro-rknn` — Kokoro v1.1-zh hybrid (default in compose)
  - `melo-rknn-zh_en` — Melo hybrid
- **Wyoming** `:10400` · **OpenAI-style API** `:10500`
- Models optional in git; bake into Docker image at build time (`bundle/`)

## Quick deploy (pull-only)

```bash
docker pull liwei19920307/sherpa-onnx-tts-stt:rknn

docker run -d --name sherpa --restart unless-stopped \
  --privileged --network host \
  -e TZ=Asia/Shanghai \
  -e PROVIDER=rknn \
  -e TTS_MODEL=kokoro-rknn -e KOKORO_VOICE=zf_001 \
  -e STT_MODEL=rknn-sense-voice \
  -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \
  -v /sys/kernel/debug:/sys/kernel/debug:ro \
  liwei19920307/sherpa-onnx-tts-stt:rknn
```

Or: `docker compose -f docker-compose.rknn.yml up -d`

Home Assistant → Integrations → **Wyoming** → `127.0.0.1` / board IP, port **10400**.

Do **not** bind-mount host app/model trees for production; the image already contains them.

More: [DEPLOY.md](./DEPLOY.md) · [RKNN.md](./RKNN.md) · [DOCS.md](./DOCS.md)

## Build image (on aarch64 board)

```bash
# Prepare bundle/tts and bundle/stt with SoC-matched .rknn packs (see DEPLOY.md)
docker build -f Dockerfile.rknn -t sherpa-onnx-tts-stt:rknn .
```

`.whl` / `.rknn` / `.onnx` are gitignored — download from upstream releases / your conversion output.

## Supported model selectors

**STT:** `rknn-sense-voice`, `rknn-paraformer-zh`, or full sherpa-onnx folder name  

**TTS:** `kokoro-rknn`, `melo-rknn-zh_en`, plus upstream CPU models (`vits-melo-tts-zh_en`, Matcha, …)

## Disclaimer

Rockchip NPU runtime (`librknnrt`) is proprietary. Converted `.rknn` files and Docker
images that include AGPL components must comply with AGPL (source offer). This project
is provided as-is for self-hosting on your own hardware.
