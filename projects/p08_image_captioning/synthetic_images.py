"""synthetic_images.py — generate a small, varied set of test images (offline).

Run:
    uv run python -m projects.p08_image_captioning.synthetic_images

Writes six deterministic images into ``./test-images`` (gitignored, so the
demo can regenerate it every run) — a landscape, a ball, a house, a text
image, an RGBA transparent PNG and a grayscale gradient — plus a
``not_an_image.txt`` to exercise the batch error path. Uses only Pillow,
already installed via the ``core`` dependency group (a transitive dependency
of matplotlib).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "test-images"


def _landscape() -> Image.Image:
    img = Image.new("RGB", (320, 200), (135, 206, 235))  # sky
    d = ImageDraw.Draw(img)
    d.rectangle([0, 150, 320, 200], fill=(60, 160, 60))  # grass
    d.ellipse([250, 20, 300, 70], fill=(255, 220, 40))  # sun
    return img


def _ball() -> Image.Image:
    img = Image.new("RGB", (240, 240), (245, 245, 245))
    ImageDraw.Draw(img).ellipse([50, 50, 190, 190], fill=(30, 90, 220))
    return img


def _house() -> Image.Image:
    img = Image.new("RGB", (300, 240), (230, 230, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([80, 130, 220, 220], fill=(200, 120, 80))  # wall
    d.polygon([(70, 130), (150, 70), (230, 130)], fill=(150, 40, 40))  # roof
    d.rectangle([135, 170, 165, 220], fill=(90, 60, 40))  # door
    return img


def _text_image() -> Image.Image:
    img = Image.new("RGB", (320, 120), (255, 255, 255))
    ImageDraw.Draw(img).text((20, 45), "HELLO WORLD", fill=(0, 0, 0))
    return img


def _transparent() -> Image.Image:
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))  # transparent background
    ImageDraw.Draw(img).ellipse([40, 40, 160, 160], fill=(220, 40, 40, 255))
    return img


def _grayscale() -> Image.Image:
    img = Image.new("L", (256, 128))
    px = img.load()
    assert px is not None  # always non-None right after Image.new
    for x in range(256):
        for y in range(128):
            px[x, y] = x  # left-right gradient
    return img


def generate(out_dir: Path = OUT) -> Path:
    """Generate the deterministic test-image set into ``out_dir``; return it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _landscape().save(out_dir / "landscape.png")
    _ball().save(out_dir / "ball.jpg")
    _house().save(out_dir / "house.png")
    _text_image().save(out_dir / "text.png")
    _transparent().save(out_dir / "transparent.png")  # RGBA
    _grayscale().save(out_dir / "gradient_gray.jpg")  # grayscale
    (out_dir / "not_an_image.txt").write_text("not an image\n", encoding="utf-8")
    return out_dir


def main() -> None:
    out_dir = generate()
    print(f"wrote test images to {out_dir}")


if __name__ == "__main__":
    main()
