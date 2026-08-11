# Rockchip (RK3566 / RK3588) NPU notes for this addon

## What can accelerate voice TTS/STT

| Hardware | Role on RK3566/RK3588 | Useful here? |
|----------|------------------------|--------------|
| **NPU (RKNPU)** | INT8/FP16 neural nets via RKNN | **Yes — STT/ASR** with `.rknn` models |
| **CPU** | Everything else | **Yes — TTS** (Matcha / Kokoro / VITS) |
| **Mali GPU** | OpenGL/Vulkan graphics | **No** — sherpa-onnx has no Mali/OpenCL provider |
| **MPP VPU** | H.264/H.265 video encode/decode | **No** — this stack is audio waveforms, not video |
| **RGA** | Image scale/convert | **No** — vision only |
| **ffmpeg** | Audio decode (mp3/wav/…) | Soft decode on CPU |

Bottom line: **NPU accelerates STT** (SenseVoice / Paraformer RKNN) **and Melo/Piper TTS decoders**. Matcha/Kokoro/full Melo-ONNX stay on CPU unless you pick a hybrid TTS model.

Upstream docs: https://k2-fsa.github.io/sherpa/onnx/rknn/index.html

## Build RKNN image (on an aarch64 board or arm64 builder)

```bash
cd sherpa-onnx-tts-stt
# Optional proxy when the board cannot reach Docker Hub / GitHub:
#   export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
docker build -f Dockerfile.rknn \
  --build-arg http_proxy --build-arg https_proxy \
  -t sherpa-onnx-tts-stt:rknn .
# Or pull: docker pull liwei19920307/sherpa-onnx-tts-stt:rknn
```

Wheels come from:
- https://k2-fsa.github.io/sherpa/onnx/rk-npu.html
- CN mirror: https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html

Host runtime known to work with many sherpa builds: **librknnrt 2.2.0**. Newer 2.3/2.4 may crash some models — prefer mounting the host library when in doubt.

## Run with NPU device

```bash
docker run --rm -it \
  --device /dev/rknpu \
  -e PROVIDER=rknn \
  -e RK_SOC=rk3588 \
  -e STT_MODEL=rknn-sense-voice \
  -e TTS_MODEL=melo-rknn-zh_en \
  -e MELO_RKNN_DIR=/tts-models/melo-rknn \
  -e LANGUAGE=zh-CN \
  -p 10400:10400 -p 10500:10500 \
  -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \
  -v /opt/sherpa/models/tts:/tts-models:ro \
  -v /opt/sherpa/models/stt:/stt-models:ro \
  sherpa-onnx-tts-stt:rknn
```

Or use `docker-compose.rknn.yml`.

Watch NPU load on the host:

```bash
sudo watch -n 0.5 cat /sys/kernel/debug/rknpu/load
```

## STT model selectors

| Config value | Behavior |
|--------------|----------|
| `rknn-sense-voice` | Auto pick `sherpa-onnx-<soc>-5-seconds-sense-voice-...` |
| `rknn-paraformer-zh` | Auto pick `sherpa-onnx-<soc>-15-seconds-paraformer-zh-...` |
| full release name | e.g. `sherpa-onnx-rk3566-5-seconds-sense-voice-zh-en-ja-ko-yue-2025-09-09` |

RKNN packs use **fixed max audio length** (5–30s). HA Assist utterances are usually short; for long audio, segment with VAD first.

`stt_thread_num` on RK3588 with `provider=rknn` selects NPU cores:
`1=AUTO`, `0=core0`, `-1=core1`, `-2=core2`, `-3=0+1`, `-4=0+1+2`.

## TTS

| Config value | Behavior |
|--------------|----------|
| **`melo-rknn-zh_en`** (default in HA) | Hybrid Melo: **encoder ONNX on CPU** + **decoder `.rknn` on NPU** (good quality) |
| **`piper-rknn-zh`** | Hybrid Piper: encoder CPU + decoder NPU (faster, lower quality) |
| `vits-melo-tts-zh_en` / matcha / kokoro | Full CPU ONNX |

Melo layout under `TTS_MODEL_DIR/melo-rknn/` (or `MELO_RKNN_DIR`):

```
encoder.onnx
decoder_rk3588.rknn   # or decoder_rk3566.rknn
g.bin
lexicon.txt
tokens.txt
```

Piper layout under `TTS_MODEL_DIR/piper-zh/` (or `PIPER_RKNN_DIR`):

```
encoder.onnx
decoder_rk*.rknn
config.json
```

On RK3588, Melo TTS uses `NPU_CORE_0_1_2` by default.

## Home Assistant addon

- Add `/dev/rknpu` in supervisor devices (already in `config.yaml`).
- Mount or bake host `librknnrt.so` (addon rarely sees host `/usr/lib` — prefer image-bundled + document bind-mount).
- Select `tts_model: melo-rknn-zh_en`, `stt_model: rknn-sense-voice`, `provider: auto|rknn`.
- Point HA Wyoming STT/TTS to this addon (ports 10400/10500).
