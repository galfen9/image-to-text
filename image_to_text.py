"""
Image-to-Text (图生文) Core Module
Supports multiple vision-language models via HuggingFace transformers.
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


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    # Florence-2 (lightweight, supports multiple tasks)
    "florence2": {
        "repo": "microsoft/Florence-2-base",
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForCausalLM,
        "trust_remote_code": True,
        "tasks": {
            "caption": "<CAPTION>",
            "detailed_caption": "<DETAILED_CAPTION>",
            "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
            "object_detection": "<OD>",
            "ocr": "<OCR>",
            "region_proposal": "<REGION_PROPOSAL>",
        },
    },
    # BLIP (classic image captioning)
    "blip": {
        "repo": "Salesforce/blip-image-captioning-base",
        "processor_cls": BlipProcessor,
        "model_cls": BlipForConditionalGeneration,
        "trust_remote_code": False,
        "tasks": {
            "caption": None,
        },
    },
    # BLIP-2 (more powerful)
    "blip2": {
        "repo": "Salesforce/blip2-opt-2.7b",
        "processor_cls": BlipProcessor,
        "model_cls": BlipForConditionalGeneration,
        "trust_remote_code": False,
        "tasks": {
            "caption": None,
        },
    },
}


class ImageToText:
    """Unified image-to-text interface supporting multiple models."""

    def __init__(
        self,
        model_name: str = "florence2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
            )

        self.model_name = model_name
        self.config = MODELS[model_name]
        self.device = device
        self.processor = None
        self.model = None
        self._cache_dir = cache_dir
        self._load()

    @property
    def available_tasks(self) -> list[str]:
        """List of available task names for this model."""
        return list(self.config["tasks"].keys())

    def _load(self):
        """Load model and processor."""
        print(f"Loading model: {self.config['repo']} ...")
        self.processor = self.config["processor_cls"].from_pretrained(
            self.config["repo"],
            trust_remote_code=self.config.get("trust_remote_code", False),
            cache_dir=self._cache_dir,
        )
        self.model = self.config["model_cls"].from_pretrained(
            self.config["repo"],
            trust_remote_code=self.config.get("trust_remote_code", False),
            cache_dir=self._cache_dir,
        )
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def describe(
        self,
        image: Image.Image,
        task: str = "caption",
        max_new_tokens: int = 512,
    ) -> str:
        """Generate text description for an image.

        Args:
            image: PIL Image.
            task: Task name (e.g. 'caption', 'detailed_caption', 'ocr').
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text description.
        """
        if task not in self.config["tasks"]:
            raise ValueError(
                f"Unknown task '{task}'. Available: {self.available_tasks}"
            )

        task_prompt = self.config["tasks"][task]

        if self.model_name == "florence2":
            return self._florence2_infer(image, task_prompt, max_new_tokens)
        elif self.model_name in ("blip", "blip2"):
            return self._blip_infer(image, max_new_tokens)
        else:
            raise NotImplementedError(f"Model '{self.model_name}' not implemented")

    def _florence2_infer(
        self, image: Image.Image, task_prompt: str, max_new_tokens: int
    ) -> str:
        """Run Florence-2 inference."""
        inputs = self.processor(
            text=task_prompt, images=image, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

        # Decode with skip_special_tokens to get clean text
        raw_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Extract task result: Florence-2 wraps in <TASK>...
        if task_prompt in raw_text:
            result = raw_text.split(task_prompt)[-1].strip()
        else:
            result = raw_text.strip()

        return result

    def _blip_infer(self, image: Image.Image, max_new_tokens: int) -> str:
        """Run BLIP inference."""
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
        )
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()

    def batch_describe(
        self,
        images: list[Image.Image],
        task: str = "caption",
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Generate descriptions for a batch of images."""
        return [
            self.describe(img, task=task, max_new_tokens=max_new_tokens)
            for img in images
        ]


# ── Convenience factory ─────────────────────────────────────────────────────
def create_model(
    model_name: str = "florence2",
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> ImageToText:
    """Create an ImageToText instance.

    Args:
        model_name: 'florence2', 'blip', or 'blip2'.
        device: 'cuda', 'cpu', or None (auto-detect).
        cache_dir: HuggingFace model cache directory.

    Returns:
        ImageToText instance.
    """
    return ImageToText(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )
