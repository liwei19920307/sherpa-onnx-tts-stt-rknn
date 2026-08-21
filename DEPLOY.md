# Docker Hub 部署（RK3588 / RK3566 / RK3568 对等）

镜像示例：`liwei19920307/sherpa-onnx-tts-stt:rknn`（可换成自建 tag）

构建时把下列内容放入 `bundle/`（见下文），镜像内两 SoC 齐全，运行时按芯片自动选：

| 组件 | RK3588 | RK356x |
|------|--------|--------|
| Kokoro decoder | `decoder_rk3588.rknn` | `decoder_rk3566.rknn` |
| Melo decoder | `decoder_rk3588.rknn` | `decoder_rk3566.rknn` |
| SenseVoice STT | `sherpa-onnx-rk3588-…` | `sherpa-onnx-rk3566-…` |

另含应用代码、misaki G2P、espeak-ng。

## 任意板子

```bash
docker pull liwei19920307/sherpa-onnx-tts-stt:rknn   # 或你的镜像名

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

`RK_SOC` 可不设。Melo：`-e TTS_MODEL=melo-rknn-zh_en`。HA Wyoming → `板子IP:10400`。

## 构建并推送

在 aarch64 机器上准备模型目录后：

```bash
cd /path/to/sherpa-onnx-tts-stt
rm -rf bundle && mkdir -p bundle/tts bundle/stt
cp -a /path/to/models/tts/kokoro-rknn bundle/tts/
cp -a /path/to/models/tts/melo-rknn bundle/tts/
cp -a /path/to/models/stt/* bundle/stt/

# 可选：HTTP 代理
# export http_proxy=... https_proxy=...
docker build --network=host \
  --build-arg http_proxy --build-arg https_proxy \
  -f Dockerfile.rknn -t YOUR_USER/sherpa-onnx-tts-stt:rknn .
docker push YOUR_USER/sherpa-onnx-tts-stt:rknn
```

模型权重与 AGPL 义务见 [NOTICE.md](./NOTICE.md)。
