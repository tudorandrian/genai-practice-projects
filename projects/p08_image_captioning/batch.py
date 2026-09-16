"""batch.py - caption every image in a folder (Project P08).

Run:
    uv run python -m projects.p08_image_captioning.batch [folder] [output_file]

Defaults: folder = ./test-images, output = ./output/captions.txt. Writes one
``filename: caption`` line per valid image; corrupt/unsupported files are
logged and skipped, never crashing the run.

Imports the sibling ``captioner`` module (not its individual names) so that
patching ``captioner.load_model`` - as the "loads the model once" test does -
is visible here too: a ``from captioner import load_model`` would bind a
private copy of the function and never see the patch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from projects.p08_image_captioning import captioner

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "test-images"
DEFAULT_OUTPUT = HERE / "output" / "captions.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def find_images(folder: Path) -> list[Path]:
    """Return sorted image files in ``folder`` (by extension)."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def caption_folder(folder: Path, output_path: Path) -> list[tuple[str, str]]:
    """Caption every image in ``folder``; write results to ``output_path``.

    Loads the BLIP model exactly once (via ``captioner.load_model``) and
    reuses it for every image in the folder, rather than relying on
    ``captioner.caption``'s per-call cache lookup. Returns the list of
    ``(filename, caption)`` pairs that succeeded; files that cannot be opened
    as images are logged and skipped.
    """
    processor, model = captioner.load_model()
    results: list[tuple[str, str]] = []
    for path in find_images(folder):
        try:
            with Image.open(path) as img:
                text = captioner.caption_image(img, processor, model)
            results.append((path.name, text))
            log.info("ok: %s: %s", path.name, text)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("skip: %s: not a valid image (%s)", path.name, exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for name, text in results:
            handle.write(f"{name}: {text}\n")
    log.info("wrote %d captions to %s", len(results), output_path)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption every image in a folder.")
    p.add_argument("folder", nargs="?", type=Path, default=DEFAULT_INPUT)
    p.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--verbose", action="store_true", help="Log at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``batch.py [folder] [output_file]``."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    if not args.folder.is_dir():
        print(f"Folder '{args.folder}' does not exist.")
        return 1
    results = caption_folder(args.folder, args.output)
    print(f"p08-image-captioning batch: {len(results)} captions")
    print(f"  wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
