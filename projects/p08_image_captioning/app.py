"""app.py - Gradio drag-and-drop image-captioning UI (Project P08).

Run:
    uv run python -m projects.p08_image_captioning.app              # http://127.0.0.1:7860
    uv run python -m projects.p08_image_captioning.app --host 0.0.0.0

The UI is a thin layer over ``captioner.caption`` - Gradio builds the whole
interface from the function signature (Image -> str), so all the real work is
the BLIP inference in ``captioner.py``. The model loads once at startup.

This module imports ``gradio`` at module level (needed to build the UI), so it
is never imported by a ``core``-marked test - see
``projects/p08_image_captioning/tests/test_captioner.py``, which only
exercises ``captioner.py`` and ``batch.py``.

Binds to ``127.0.0.1`` by default; pass ``--host 0.0.0.0`` to listen on every
interface (e.g. to reach the UI from another machine on the network).
"""

from __future__ import annotations

import argparse

import gradio as gr
from PIL import Image

from projects.p08_image_captioning.captioner import MODEL_NAME, caption, load_model

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


def describe_image(image: Image.Image | None) -> str:
    """Gradio callback: caption an uploaded image, with a friendly error message."""
    if image is None:
        return "Upload an image first."
    try:
        return caption(image)
    except Exception as exc:  # never crash the server on a bad upload
        return f"Could not process the image: {exc}"


def build_demo() -> gr.Interface:
    """Construct the Gradio interface (kept out of import side effects)."""
    return gr.Interface(
        fn=describe_image,
        inputs=gr.Image(type="pil", label="Upload an image (drag-and-drop)"),
        outputs=gr.Textbox(label="Generated caption"),
        title="Image Captioning with BLIP",
        description=(
            f"The {MODEL_NAME} model describes the uploaded image. "
            "Runs locally on CPU - the first run downloads the model (~1 GB)."
        ),
        flagging_mode="never",  # Gradio 5+ name (was allow_flagging in Gradio 4)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the BLIP captioning UI.")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST}).")
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the model once, then start the Gradio server."""
    args = parse_args(argv)
    load_model()  # warm the cache before serving so the first request is fast
    build_demo().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
