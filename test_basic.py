#!/usr/bin/env python3
"""
Quick smoke test: generate a test image and run through the model.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Create a test image with simple shapes
def make_test_image():
    img = Image.new("RGB", (400, 300), color="lightblue")
    draw = ImageDraw.Draw(img)

    # Draw a simple house
    draw.rectangle([150, 120, 250, 250], fill="brown", outline="darkred", width=2)
    draw.polygon([(140, 120), (200, 50), (260, 120)], fill="red", outline="darkred")
    draw.rectangle([180, 170, 220, 250], fill="#654321", outline="black")
    draw.rectangle([160, 140, 180, 170], fill="#FFD700", outline="#FF8C00")
    draw.rectangle([220, 140, 240, 170], fill="#FFD700", outline="#FF8C00")

    # Draw sun
    draw.ellipse([320, 30, 370, 80], fill="yellow", outline="orange")

    # Draw grass
    draw.rectangle([0, 250, 400, 300], fill="green")

    # Draw text
    try:
        font = ImageFont.truetype("/System Fonts/Arial.ttf", 24)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text([20, 20], "Test Image", fill="black", font=font)

    return img


def main():
    test_img = make_test_image()
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    test_img.save(output_dir / "test_house.png")
    print(f"Test image saved to {output_dir / 'test_house.png'}")

    # Run inference
    from image_to_text import create_model

    print("\n--- Testing Florence-2 ---")
    model = create_model(model_name="florence2")
    for task in model.available_tasks[:3]:  # Test first 3 tasks
        try:
            result = model.describe(test_img, task=task)
            print(f"Task [{task}]: {result[:200]}...")
        except Exception as e:
            print(f"Task [{task}] ERROR: {e}")

    print("\n--- Testing BLIP ---")
    model_blip = create_model(model_name="blip")
    result = model_blip.describe(test_img, task="caption")
    print(f"Task [caption]: {result}")

    print("\n✅ Smoke test complete!")
    return 0


if __name__ == "__main__":
    main()
