"""captioner.py — BLIP image-captioning inference, decoupled from any UI.

Project P08. This module is the *engine*: it loads the BLIP model once and
turns a PIL image into an English caption. Both entry points — ``app.py``
(Gradio) and ``batch.py`` (folder -> captions.txt) — import from here, so the
model is never loaded twice and the UI layer stays swappable. ``caption_image``
is pure from the caller's side (Image -> str) and accepts an injected
processor/model, so it is testable without downloading the ~1 GB model or
starting a server.

The heavy imports (torch, transformers) are deferred into ``device`` and
``load_model`` so this module — and its pure logic, unit-tested with fakes —
stays importable on a machine where those packages are not installed (only
the ``models`` dependency group needs them).

Run
    uv run p08-image-captioning --demo             # offline-ish demo, writes output/
    uv run p08-image-captioning --image path.png    # caption a single image
    uv run pytest projects/p08_image_captioning -q  # fast tests, no weights needed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 50

# Module-level singletons so the weights load exactly once per process.
_PROCESSOR: Any = None
_MODEL: Any = None


def device() -> str:
    """Return ``"cuda"`` if a CUDA GPU is available, else ``"cpu"``.

    Imports torch lazily so this function — and this module as a whole — is
    importable and testable (with a monkeypatched ``sys.modules["torch"]`` or
    without torch installed at all) without the ``models`` dependency group.
    """
    import torch  # lazy heavy import, see module docstring

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = MODEL_NAME) -> tuple[Any, Any]:
    """Load (once) and return the BLIP ``(processor, model)`` pair on ``device()``.

    The first call downloads the weights from the Hugging Face Hub into the
    local cache (``~/.cache/huggingface``); later calls reuse the in-process
    singletons. Imports transformers lazily so this module stays importable
    without it.
    """
    global _PROCESSOR, _MODEL
    if _PROCESSOR is None or _MODEL is None:
        # Lazy heavy import so this module is importable/testable without transformers.
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
        )

        _PROCESSOR = BlipProcessor.from_pretrained(model_name)
        # Explicitly `Any`, not the inferred `BlipForConditionalGeneration`: with
        # transformers installed, its `from_pretrained`/`nn.Module.to()` overloads
        # make mypy reject `.to(device())` (a str) below; without transformers
        # installed (`ignore_missing_imports` makes the import `Any`), that same
        # call needs no such ignore, so a fixed `# type: ignore` would be flagged
        # as unused in the other environment. Widening the type here avoids the
        # overload check altogether, so no ignore comment is needed in either case.
        blip_model: Any = BlipForConditionalGeneration.from_pretrained(model_name)
        _MODEL = blip_model.to(device())
    return _PROCESSOR, _MODEL


def caption_image(
    image: Image.Image, processor: Any, model: Any, max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Return an English caption for ``image`` (pure: Image -> str).

    Converts to RGB first so RGBA/grayscale inputs work. ``processor`` and
    ``model`` are injected, which keeps this function free of global state and
    unit-testable with fakes. Generation is greedy and deterministic
    (``num_beams=1, do_sample=False``) so the same image always yields the
    same caption. Raises ``TypeError`` if ``image`` is not a PIL image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    text = processor.decode(output[0], skip_special_tokens=True)
    return str(text).strip()


def caption(image: Image.Image, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Convenience wrapper: caption ``image`` using the cached BLIP model."""
    processor, model = load_model()
    return caption_image(image, processor, model, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# Demo, CLI
# ---------------------------------------------------------------------------


def demo() -> DemoResult:
    """Caption the six synthetic test images with real BLIP weights.

    Tier ``models``: downloads/loads the real model and never runs in CI.
    Generates the synthetic image set, captions every image once (the model
    loads exactly once — see ``batch.caption_folder``), and writes a
    deterministic ``output/captions.txt`` and ``output/metrics.txt``. Only the
    console summary and the returned ``DemoResult`` carry timing; the two
    committed files never vary between runs (greedy decoding is deterministic).
    """
    start = time.perf_counter()

    from projects.p08_image_captioning import batch, synthetic_images

    synthetic_images.generate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = batch.caption_folder(synthetic_images.OUT, OUT_DIR / "captions.txt")

    selected_device = device()
    metrics_lines = [
        f"images={len(rows)}",
        f"device={selected_device}",
        f"model={MODEL_NAME}",
    ]
    (OUT_DIR / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    seconds = time.perf_counter() - start
    figures = {"images": str(len(rows)), "device": selected_device, "model": MODEL_NAME}
    log.info("demo: captioned %d images on %s", len(rows), selected_device)
    return DemoResult("p08-image-captioning", "ok", figures, seconds=round(seconds, 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--image", type=Path, help="Caption a single image file and print it.")
    p.add_argument("--demo", action="store_true", help="Run the demo (needs the models group).")
    p.add_argument("--verbose", action="store_true", help="Log at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.image:
        with Image.open(args.image) as img:
            text = caption(img)
        print(f"p08-image-captioning: {args.image.name}")
        print(f"  caption: {text}")
        return 0

    if args.demo:
        result = demo()
        print(f"p08-image-captioning: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    print("Nothing to do: pass --demo, --image PATH, or run the app/batch modules directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
