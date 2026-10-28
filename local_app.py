"""
图生文 (Image-to-Text) - 本地版
使用 MPS (Apple Silicon) 加速，完全本地运行，无需联网。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipForConditionalGeneration,
    BlipProcessor,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    # Florence-2 (多任务，推荐)
    "florence2": {
        "repo": "microsoft/Florence-2-base",
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForCausalLM,
        "trust_remote_code": True,
        "tasks": {
            "caption": "caption",
            "detailed_caption": "detailed_caption",
            "more_detailed_caption": "more_detailed_caption",
            "ocr": "ocr",
            "region_proposal": "region_proposal",
        },
    },
    # BLIP (经典描述)
    "blip": {
        "repo": "Salesforce/blip-image-captioning-base",
        "processor_cls": BlipProcessor,
        "model_cls": BlipForConditionalGeneration,
        "trust_remote_code": False,
        "tasks": {"caption": None},
    },
}


class ImageToText:
    """本地图生文引擎 (MPS / CPU)。"""

    def __init__(
        self,
        model_name: str = "florence2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model. Available: {list(MODELS.keys())}")
        self.model_name = model_name
        self.config = MODELS[model_name]
        self._device = device or DEVICE
        self._cache_dir = cache_dir
        self._proc = None
        self._model = None
        self._load()

    @property
    def device(self) -> str:
        return self._device

    @property
    def available_tasks(self) -> list[str]:
        return list(self.config["tasks"].keys())

    def _load(self):
        print(f"[{self.model_name}] Loading on {self._device} ...")
        self._proc = self.config["processor_cls"].from_pretrained(
            self.config["repo"],
            trust_remote_code=self.config["trust_remote_code"],
            cache_dir=self._cache_dir,
        )
        self._model = self.config["model_cls"].from_pretrained(
            self.config["repo"],
            trust_remote_code=self.config["trust_remote_code"],
            cache_dir=self._cache_dir,
        ).to(self._device)
        self._model.eval()
        print(f"[{self.model_name}] Ready ({self._device})")

    def describe(
        self,
        image: Image.Image,
        task: str = "caption",
        max_new_tokens: int = 512,
    ) -> str:
        if task not in self.config["tasks"]:
            raise ValueError(f"Task '{task}' not available. Use: {self.available_tasks}")
        task_key = self.config["tasks"][task]
        return self._florence2_infer(image, task_key, max_new_tokens) if self.model_name == "florence2" else self._blip_infer(image, max_new_tokens)

    def _florence2_infer(self, image: Image.Image, task_key: str, max_new_tokens: int) -> str:
        task_token = f"<{task_key.upper()}>"
        inputs = self._proc(text=task_token, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self._proc.batch_decode(out, skip_special_tokens=True)[0]
        return text.replace(task_token, "").strip() if task_token in text else text.strip()

    def _blip_infer(self, image: Image.Image, max_new_tokens: int) -> str:
        inputs = self._proc(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=5)
        return self._proc.decode(out[0], skip_special_tokens=True).strip()

    def close(self):
        """释放 GPU 内存。"""
        model = getattr(self, '_model', None)
        proc = getattr(self, '_proc', None)
        if model is not None:
            del self._model, self._proc
            if self._device == "mps":
                torch.mps.empty_cache()

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass  # during interpreter shutdown


# ── Factory ─────────────────────────────────────────────────────────────────
def create_model(model_name: str = "florence2", device: Optional[str] = None, cache_dir: Optional[str] = None) -> ImageToText:
    return ImageToText(model_name=model_name, device=device, cache_dir=cache_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys, json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="图生文 (本地 MPS/CPU)")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--task", choices=MODELS["florence2"]["tasks"].keys(), default="caption")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="florence2")
    args = parser.parse_args()

    model = create_model(args.model)
    img = Image.open(args.image).convert("RGB")
    result = model.describe(img, task=args.task)
    print(f"[{args.task}] {result}")
    model.close()
