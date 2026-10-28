#!/usr/bin/env python3
"""
统一媒体生成工具 — 图生文 / 文生图 / 文生视频
每次运行前选择：本地 (MPS) 或 云端 (Comfy Cloud)

用法:
  python media_gen.py image   --prompt "描述"              # 生成图片
  python media_gen.py text    --image photo.jpg            # 图片转文字
  python media_gen.py video   --prompt "描述"              # 生成视频
  python media_gen.py         --prompt "描述"              # 智能选择 (默认图片)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
LOCAL_PY = PROJECT_DIR / "local_app.py"
VIDEO_PY = PROJECT_DIR / "local_video.py"


def ask_local_or_cloud(kind: str) -> str:
    """询问用户选择本地还是云端。"""
    if kind == "video":
        print(f"\n{'='*50}")
        print(f"  🎬 视频生成 — 选择方式")
        print(f"{'='*50}")
        print(f"  1. 本地 (MPS)  — 免费, 离线, 但慢 (约 3-5 分钟)")
        print(f"  2. 云端 (Comfy) — 快 (几秒), 需联网, 可能有额度限制")
    else:
        print(f"\n{'='*50}")
        print(f"  🖼️  {kind}生成 — 选择方式")
        print(f"{'='*50}")
        print(f"  1. 本地 (MPS)  — 免费, 离线, 秒级响应")
        print(f"  2. 云端 (Comfy) — 快 (几秒), 需联网, 可能有额度限制")

    print(f"{'='*50}\n")
    choices = {"1": "local", "2": "cloud", "": "cloud"}

    while True:
        try:
            choice = input("选择 (1=本地, 2=云端, 默认=云端): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""
        if choice in choices:
            return choices[choice]
        print("  请输入 1 或 2")


def run_local_text(image_path: str) -> None:
    """本地图生文。"""
    import subprocess
    cmd = [
        sys.executable, str(LOCAL_PY), image_path,
        "--task", "caption",
    ]
    print("\n运行本地图生文...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        # 打印输出中最后一行（实际结果）
        for line in result.stdout.strip().split("\n"):
            if "Loading" in line or "Ready" in line:
                continue
            if line.strip():
                print(line)
    else:
        print(f"错误: {result.stderr[-200:]}")


def run_local_video(prompt: str, frames: int, steps: int, output: str) -> None:
    """本地文生视频。"""
    import subprocess
    cmd = [
        sys.executable, str(VIDEO_PY), prompt,
        "--frames", str(frames),
        "--steps", str(steps),
        "--output", output,
    ]
    print(f"\n运行本地视频生成... (预计 3-5 分钟)")
    result = subprocess.run(cmd, timeout=600)
    if result.returncode != 0:
        print("视频生成失败，请检查可用内存。")


def run_cloud(kind: str, prompt: str, image_path: str = None) -> None:
    """
    云端生成 — 通过 Comfy Cloud MCP 生成。
    这个由 AI 代理直接调用工具完成，不需要脚本。
    """
    # 云端生成由 AI 代理调用 Comfy Cloud MCP 工具完成
    # 这里只是提示用户
    if kind == "video":
        print("\n云端视频生成需要 AI 代理调用 Comfy Cloud MCP 工具。")
        print("请在对话中告诉我：'帮我生成一段视频：' + 描述")
    elif kind == "image":
        print("\n云端图片生成需要 AI 代理调用 Comfy Cloud MCP 工具。")
        print("请在对话中告诉我：'帮我生成一张图片：' + 描述")
    elif kind == "text":
        print("\n云端图生文需要 AI 代理调用 HuggingFace API。")
        print("请在对话中告诉我：'帮我描述这张图片' + 上传图片")


def main():
    parser = argparse.ArgumentParser(
        description="统一媒体生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["image", "text", "video"],
        help="生成类型: image(文生图), text(图生文), video(文生视频)",
    )
    parser.add_argument("--prompt", "-p", help="描述文本 (图片/视频生成用)")
    parser.add_argument("--image", "-i", help="输入图片 (图生文用)")
    parser.add_argument("--frames", type=int, default=4, help="视频帧数 (默认 4)")
    parser.add_argument("--steps", type=int, default=5, help="推理步数 (默认 5)")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--local", action="store_true", help="强制本地生成")
    parser.add_argument("--cloud", action="store_true", help="强制云端生成")

    args = parser.parse_args()

    # 参数校验
    if args.mode == "text" and not args.image:
        parser.error("图生文需要 --image 参数")
    if args.mode in ("image", "video") and not args.prompt:
        parser.error(f"{args.mode} 生成需要 --prompt 参数")

    # 选择本地或云端
    if args.local:
        method = "local"
    elif args.cloud:
        method = "cloud"
    else:
        method = ask_local_or_cloud(args.mode)

    # 执行
    if method == "local":
        if args.mode == "text":
            run_local_text(args.image)
        elif args.mode == "video":
            out = args.output or PROJECT_DIR / f"video_{args.mode}.mp4"
            run_local_video(args.prompt, args.frames, args.steps, str(out))
        elif args.mode == "image":
            print("\n⚠️ 本地图片生成暂不支持，请使用云端。")
    else:
        run_cloud(args.mode, args.prompt, args.image)


if __name__ == "__main__":
    main()
