# Home Assistant Add-on: Sherpa Onnx TTS/STT

![Supports aarch64 Architecture][aarch64-shield] ![Supports amd64 Architecture][amd64-shield]

Supports Kokoro-TTS!!

Offline Sherpa-onnx TTS/STT with wyoming support, supports kokoro-TTS/matcha-TTS/paraformer-STT, requires 1.5GB RAM.

离线Sherpa-onnx TTS/STT的wyoming集成，支持kokoro-TTS/matcha-TTS/paraformer-STT，需要1.5G内存。

Also supports Openai-format TTS/STT api  IP:10500/v1/audio/speech IP:10500/v1/audio/transcriptions

同时支持Openai TTS/STT 格式两个接口  IP:10500/v1/audio/speech IP:10500/v1/audio/transcriptions
(It just works. PR is welcomed to improve this.)

## Rockchip NPU (RK3566 / RK3588)

This fork can run **STT on the Rockchip NPU** via sherpa-onnx `provider=rknn` and official `.rknn` ASR packs.

| Block | Accelerates this addon? |
|-------|-------------------------|
| NPU | Yes — SenseVoice / Paraformer STT |
| CPU | Yes — all TTS (Matcha/Kokoro/VITS) |
| Mali GPU | No |
| MPP hard decode / RGA | No (video only) |

See **[RKNN.md](./RKNN.md)** and `docker-compose.rknn.yml`.

```bash
docker build --build-arg BUILD_TYPE=rknn -t sherpa-onnx-tts-stt:rknn .
docker compose -f docker-compose.rknn.yml up
```

## Supported STT Models:
* rknn-sense-voice (auto chip-matched SenseVoice `.rknn` on NPU)
* rknn-paraformer-zh (auto chip-matched Paraformer-zh `.rknn` on NPU)
* sherpa-onnx-paraformer-zh-2023-03-28 (Chinese Only, CPU / CUDA)
* sherpa-onnx-paraformer-zh-small-2024-03-09 (Chinese Only, CPU / CUDA)

## Supported TTS Models:
* matcha-icefall-zh-baker (Chinese Only)
* vits-melo-tts-zh_en (Chinese and English)
* kokoro-int8-multi-lang-v1_1 (Multiple-Languages)

## Custom Models are supported.
See [DOCS.md](./DOCS.md) for documentation.

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
