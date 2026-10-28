# 图生文项目 (Image-to-Text)

本地 + 云端混合方案，适用于 Apple Silicon Mac (MPS)。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 本地图生文 (MPS 加速, 完全离线)
python local_app.py photo.jpg           # 简洁描述
python local_app.py photo.jpg --task detailed_caption   # 详细描述
python local_app.py photo.jpg --task ocr               # 提取文字

# 3. 本地文生视频 (MPS, 需要 >=8GB 可用 RAM)
python local_video.py "a cat walking in a park" -o video.mp4
```

## 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| 系统 | macOS 14+ (Apple Silicon) | macOS 15+ |
| RAM | 16GB | 32GB+ |
| GPU | MPS (Metal) | 16GB+ VRAM |
| 可用内存 | 图生文: 4GB+ | 视频: 8GB+ |

## 可用模型

### 图生文 (image_to_text.py)
- **Florence-2** (推荐): 多任务 - 描述/详细描述/OCR/区域建议
- **BLIP**: 经典图片描述

### 文生视频 (local_video.py)
- **ModelScope T2V**: ~2GB, 适合 8GB+ RAM
- **LTX-Video 2B**: ~4GB, 需要 16GB+ RAM (更高画质)

## 两种模式

| 特性 | 本地模式 (local_app.py) | 云端模式 (app.py) |
|------|------------------------|------------------|
| 联网 | ❌ 不需要 | ✅ 需要 |
| 隐私 | 完全本地 | 上传到 HuggingFace |
| 速度 | 取决于硬件 (MPS) | 云端GPU加速 |
| 画质 | Florence-2 / BLIP | FLUX / MOSS / Wan2.1 |

## 项目结构

```
image-to-text/
├── local_app.py          ← 本地图生文 (MPS)
├── local_video.py        ← 本地文生视频 (MPS)
├── image_to_text.py      ← 云端图生文 (HuggingFace)
├── cli.py                ← 云端 CLI 工具
├── app.py                ← 云端 Web UI
├── requirements.txt      ← 依赖列表
└── README.md             ← 本文档
```

## 注意事项

1. **首次运行**会下载模型权重 (~300MB - 2GB)，之后缓存到本地
2. **视频生成**需要较大内存，生成前建议关闭其他应用
3. Apple Silicon 上运行 MPS 模型比 CPU 快 5-10 倍
