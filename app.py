#!/usr/bin/env python3
"""
Gradio web UI for Image-to-Text.

Usage:
    python app.py          # starts on http://localhost:7860
    python app.py --server 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
import gradio as gr

from image_to_text import create_model, MODELS


class App:
    """Gradio-based web UI."""

    def __init__(self, model_name: str = "florence2"):
        self.model_name = model_name
        self.model = create_model(model_name=model_name)

    def generate(
        self,
        image: Image.Image | None,
        task: str,
        max_tokens: int,
    ) -> str:
        if image is None:
            return "Please upload an image."
        result = self.model.describe(image, task=task, max_new_tokens=max_tokens)
        return result or "(No output generated)"

    def build_ui(self) -> gr.Blocks:
        tasks = self.model.available_tasks

        with gr.Blocks(
            title="图生文 (Image-to-Text)",
            theme=gr.themes.Soft(primary_hue="blue"),
        ) as ui:
            gr.Markdown(
                """
                # 🖼️ 图生文 (Image-to-Text)
                Upload an image to generate text descriptions using vision-language models.

                **Supported models:** Florence-2, BLIP, BLIP-2
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=300,
                    )
                    with gr.Row():
                        model_sel = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value=self.model_name,
                            label="Model",
                        )
                        task_sel = gr.Dropdown(
                            choices=tasks,
                            value=tasks[0],
                            label="Task",
                        )
                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=2048,
                        value=512,
                        step=32,
                        label="Max Tokens",
                    )
                    btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output = gr.Textbox(
                        label="Generated Text",
                        lines=6,
                        interactive=False,
                    )
                    example_images = gr.Gallery(
                        label="Example",
                        columns=3,
                        height=200,
                    )

            # Task change updates task selector
            def on_model_change(model_name):
                model = create_model(model_name=model_name)
                return gr.Dropdown(
                    choices=model.available_tasks,
                    value=model.available_tasks[0],
                    interactive=True,
                )

            model_sel.change(
                fn=on_model_change,
                inputs=model_sel,
                outputs=task_sel,
            )

            btn.click(
                fn=self.generate,
                inputs=[img_input, task_sel, max_tokens],
                outputs=output,
            )

            # Keyboard shortcut: Enter to generate
            def generate_on_enter(img, task, tokens):
                return self.generate(img, task, tokens)

            img_input.submit(
                fn=generate_on_enter,
                inputs=[img_input, task_sel, max_tokens],
                outputs=output,
            )

        return ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-to-Text Web UI")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="florence2")
    parser.add_argument("--server", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    return parser.parse_args()


def main():
    args = parse_args()
    app = App(model_name=args.model)
    ui = app.build_ui()
    print(f"Starting Image-to-Text UI at http://{args.server}:{args.port}")
    ui.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
