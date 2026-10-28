#!/usr/bin/env python3
"""
Command-line interface for Image-to-Text.

Usage:
    python cli.py path/to/image.jpg                        # caption
    python cli.py path/to/image.jpg --task detailed_caption
    python cli.py path/to/image.jpg --model blip
    python cli.py --dir path/to/images/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from image_to_text import create_model, MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image-to-Text: generate text descriptions from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Path to input image file",
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        help="Process all images in a directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (JSON). Default: stdout",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default="florence2",
        help="Vision-language model to use (default: florence2)",
    )
    parser.add_argument(
        "--task", "-t",
        choices=["caption", "detailed_caption", "more_detailed_caption", "object_detection", "ocr", "region_proposal"],
        default="caption",
        help="Task type (default: caption)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="HuggingFace model cache directory",
    )
    return parser.parse_args()


def process_single(
    image_path: Path,
    model: ImageToText,
    task: str,
    max_tokens: int,
) -> dict:
    """Process a single image."""
    print(f"Processing: {image_path}")
    img = Image.open(image_path).convert("RGB")
    result = model.describe(img, task=task, max_new_tokens=max_tokens)
    return {"image": str(image_path), "task": task, "text": result}


def process_directory(
    dir_path: Path,
    model: ImageToText,
    task: str,
    max_tokens: int,
) -> list[dict]:
    """Process all images in a directory."""
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_exts
    )
    if not image_files:
        print(f"No images found in {dir_path}")
        return []

    print(f"Found {len(image_files)} images in {dir_path}")
    results = []
    for i, img_path in enumerate(image_files, 1):
        try:
            r = process_single(img_path, model, task, max_tokens)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {img_path}: {e}", file=sys.stderr)
    return results


def main():
    args = parse_args()

    # Validate input
    if not args.image and not args.dir:
        print("Error: specify an image file or --dir", file=sys.stderr)
        sys.exit(1)

    # Resolve device
    device = None if args.device == "auto" else args.device

    # Create model
    model = create_model(
        model_name=args.model,
        device=device,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )

    # Process
    if args.image:
        if not args.image.is_file():
            print(f"Error: file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        results = [process_single(args.image, model, args.task, args.max_tokens)]
    else:
        results = process_directory(args.dir, model, args.task, args.max_tokens)

    # Output
    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"Results written to {args.output}")
    else:
        for r in results:
            print(f"\n{'=' * 60}")
            print(f"Image: {r['image']}")
            print(f"Task:  {r['task']}")
            print(f"Text:  {r['text']}")

    return 0


if __name__ == "__main__":
    main()
