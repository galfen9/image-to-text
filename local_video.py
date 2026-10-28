"""
文生视频 (Text-to-Video) - 本地版
使用 diffusers + CogVideoX-2b + MPS (Apple Silicon)。

⚠️ 性能: 4帧5步 ~3分钟 (MPS), 6帧10步 ~5-8分钟
✅ 模型已缓存: 3.5GB 本地可用
✅ 完全本地, 不联网, 免费

用法:
  python local_video.py "a cat walking" -o video.mp4
  python local_video.py "sunset ocean" --frames 4 --steps 5
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from diffusers import CogVideoXPipeline

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 本地缓存的模型路径
MODEL_PATH = "/Users/a123/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"


class VideoGenerator:
    """本地视频生成 (CogVideoX-2b, MPS/CPU)。"""

    def __init__(self, device: Optional[str] = None):
        self._device = device or DEVICE
        self._pipe = None
        self._load()

    @property
    def device(self) -> str:
        return self._device

    def _load(self):
        print(f"Loading CogVideoX-2b on {self._device} ...")
        self._pipe = CogVideoXPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16
        )
        # MPS 不支持 float64
        self._pipe = self._pipe.to(torch.float32)
        self._pipe = self._pipe.to(self._device)
        print(f"Ready ({self._device})")

    def generate(
        self,
        prompt: str,
        num_frames: int = 4,
        num_inference_steps: int = 5,
        output_path: Optional[str] = None,
    ) -> Path:
        """生成视频。推荐 4-6 帧, 5-10 步。"""
        print(f"Generating: {prompt[:60]}...")
        video = self._pipe(
            prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,
        )

        if output_path is None:
            output_path = f"video_{int(torch.randint(0, 999999, (1,)).item())}.mp4"

        frames = video.frames[0]
        frames_u8 = [(f.permute(1,2,0).numpy() * 255).astype("uint8") for f in frames]
        import imageio

        imageio.mimwrite(output_path, frames_u8, fps=8, quality=5)
        print(f"Saved: {output_path}")
        return Path(output_path)

    def close(self):
        pipe = getattr(self, '_pipe', None)
        if pipe is not None:
            del self._pipe
            if self._device == "mps":
                torch.mps.empty_cache()

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文生视频 (本地 MPS)")
    parser.add_argument("prompt", help="视频描述")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    try:
        gen = VideoGenerator()
        gen.generate(args.prompt, num_frames=args.frames, num_inference_steps=args.steps, output_path=args.output)
        gen.close()
    except Exception as e:
        print(f"Error: {e}")
