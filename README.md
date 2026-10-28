# 🖼️ 图生文 (Image-to-Text)

基于 HuggingFace 视觉语言模型的图片生成文字项目。支持多种预训练模型，提供 CLI 和 Web UI。

## 功能特性

- **多模型支持**: Florence-2、BLIP、BLIP-2
- **多种任务**: 图片描述、详细描述、OCR 文字提取、目标检测
- **命令行工具**: 单文件或批量目录处理
- **Web UI**: Gradio 可视化界面，支持实时预览

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 命令行使用

```bash
# 单张图片描述
python cli.py photo.jpg

# 详细描述
python cli.py photo.jpg --task detailed_caption

# 使用 BLIP 模型
python cli.py photo.jpg --model blip

# 批量处理目录
python cli.py --dir ./photos/ --output results.json

# 提取图片中的文字 (OCR)
python cli.py photo.jpg --task ocr
```

### Web UI

```bash
python app.py              # 本地访问 http://localhost:7860
python app.py --share      # 生成公网分享链接
python app.py --model blip  # 使用不同模型
```

## 支持的模型

| 模型 | 大小 | 速度 | 能力 |
|------|------|------|------|
| Florence-2 | ~300MB | 快 | 描述、OCR、检测、区域建议 |
| BLIP | ~400MB | 中 | 图片描述 |
| BLIP-2 | ~5GB | 慢 | 高质量图片描述 |

## 任务类型

| 任务 | 说明 |
|------|------|
| `caption` | 简洁描述图片内容 |
| `detailed_caption` | 详细描述 |
| `more_detailed_caption` | 更详细的描述 |
| `ocr` | 提取图片中的文字 |
| `object_detection` | 检测图片中的物体 |
| `region_proposal` | 提出感兴趣区域 |

## 项目结构

```
image-to-text/
├── image_to_text.py   # 核心模块
├── cli.py             # 命令行接口
├── app.py             # Gradio Web UI
├── test_basic.py      # 冒烟测试
├── requirements.txt   # 依赖
├── README.md          # 文档
└── output/            # 测试输出目录
```

## 系统要求

- Python 3.10+
- 推荐有 GPU（CUDA）以加速推理
- CPU 也可运行，速度较慢
