# Rockchip (RK3566 / RK3588) NPU notes

## What can accelerate voice TTS/STT

| Hardware | Role | Useful here? |
|----------|------|--------------|
| **NPU (RKNPU)** | `.rknn` nets | **Yes** — STT + Kokoro/Melo/Piper **decoders** |
| **CPU** | ONNX encoder / G2P / full CPU TTS | **Yes** |
| **Mali GPU** | Graphics | **No** |
| **MPP / RGA** | Video / image | **No** |

Upstream ASR docs: https://k2-fsa.github.io/sherpa/onnx/rknn/index.html

## Run (production)

Models and app code are **inside the image**. Only mount the host NPU runtime:

```bash
docker run -d --name sherpa --restart unless-stopped \
  --privileged --network host \
  -e PROVIDER=rknn \
  -e TTS_MODEL=kokoro-rknn -e KOKORO_VOICE=zf_001 \
  -e STT_MODEL=rknn-sense-voice \
  -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \
  -v /sys/kernel/debug:/sys/kernel/debug:ro \
  YOUR_USER/sherpa-onnx-tts-stt:rknn
```

See `docker-compose.rknn.yml` and [DEPLOY.md](./DEPLOY.md).

Prefer **librknnrt 2.2.0** on the host for many sherpa RKNN builds.

## STT selectors

| Config | Behavior |
|--------|----------|
| `rknn-sense-voice` | Auto `sherpa-onnx-<soc>-5-seconds-sense-voice-…` |
| `rknn-paraformer-zh` | Auto Paraformer-zh RKNN pack |
| full folder name | Exact pack under `STT_MODEL_DIR` |

RK3588 `stt_thread_num` with `provider=rknn`: `1=AUTO`, `0/ -1/ -2` = core0/1/2, `-3=0+1`, `-4=0+1+2`.

## TTS

| Config | Behavior |
|--------|----------|
| `kokoro-rknn` | Encoder/har CPU + decoder NPU |
| `melo-rknn-zh_en` | Encoder CPU + decoder NPU (**AGPL-derived** helpers — see NOTICE.md) |
| `piper-rknn-zh` | Optional hybrid Piper |
| `vits-melo-tts-zh_en` / Matcha / kokoro ONNX | Full CPU |

Kokoro layout (`TTS_MODEL_DIR/kokoro-rknn/`):

```
encoder.onnx
har_generator.onnx
decoder_rk3588.rknn
decoder_rk3566.rknn
config.json
voices/*.npy
```

Melo layout (`TTS_MODEL_DIR/melo-rknn/`):

```
encoder.onnx
decoder_rk3588.rknn
decoder_rk3566.rknn
g.bin
lexicon.txt
tokens.txt
```

## Build notes

- Place packs in `bundle/tts` + `bundle/stt` before `docker build -f Dockerfile.rknn`.
- Do not commit `.whl` / `.rknn` / `.onnx` (see `.gitignore`).
- Conversion helpers: [tools/README.md](./tools/README.md).
