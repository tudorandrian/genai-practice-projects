"""Tests for captioner.py and batch.py.

The fast tests inject a fake processor/model, so ``caption_image`` and
``caption_folder`` are exercised without downloading BLIP or importing torch.
One test actually runs real BLIP inference and is marked ``models`` - it is
excluded from the CI marker filter (``core and not network``) and only runs
when the ``models`` dependency group is installed.

Run:
    uv run pytest projects/p08_image_captioning -q            # fast, core only
    uv run pytest projects/p08_image_captioning -m models -q  # real weights
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from projects.p08_image_captioning import batch, captioner

# No blanket module-level ``pytestmark`` here: this file mixes fast fake-model
# tests with one real-BLIP test, and a module-level marker would add "core" to
# every test including the real-BLIP one - which the CI filter
# (``core and not network``) would then wrongly pick up and run without the
# ``models`` dependency group installed. Each fast test is marked
# individually instead.


class FakeProcessor:
    """Records the mode of the image it receives; returns canned decode text."""

    def __init__(self) -> None:
        self.last_mode: str | None = None

    def __call__(self, images: Image.Image, return_tensors: str | None = None) -> dict[str, Any]:
        self.last_mode = images.mode
        return {"pixel_values": images}

    def decode(self, _token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "  a fake caption  "  # padded to prove caption_image strips it


class FakeModel:
    def __init__(self) -> None:
        self.max_new_tokens: int | None = None

    def generate(self, max_new_tokens: int | None = None, **_inputs: Any) -> list[list[int]]:
        self.max_new_tokens = max_new_tokens
        return [[0, 1, 2, 3]]


# =============================================================================
# caption_image - pure
# =============================================================================


@pytest.mark.core
def test_returns_stripped_caption() -> None:
    proc, model = FakeProcessor(), FakeModel()
    out = captioner.caption_image(Image.new("RGB", (8, 8)), proc, model)
    assert out == "a fake caption"


@pytest.mark.core
def test_converts_rgba_to_rgb() -> None:
    proc, model = FakeProcessor(), FakeModel()
    captioner.caption_image(Image.new("RGBA", (8, 8)), proc, model)
    assert proc.last_mode == "RGB"


@pytest.mark.core
def test_converts_grayscale_to_rgb() -> None:
    proc, model = FakeProcessor(), FakeModel()
    captioner.caption_image(Image.new("L", (8, 8)), proc, model)
    assert proc.last_mode == "RGB"


@pytest.mark.core
def test_max_new_tokens_is_passed_through() -> None:
    proc, model = FakeProcessor(), FakeModel()
    captioner.caption_image(Image.new("RGB", (8, 8)), proc, model, max_new_tokens=17)
    assert model.max_new_tokens == 17


@pytest.mark.core
def test_default_max_new_tokens() -> None:
    proc, model = FakeProcessor(), FakeModel()
    captioner.caption_image(Image.new("RGB", (8, 8)), proc, model)
    assert model.max_new_tokens == captioner.MAX_NEW_TOKENS


@pytest.mark.core
def test_type_error_on_non_image() -> None:
    with pytest.raises(TypeError):
        captioner.caption_image("not an image", FakeProcessor(), FakeModel())  # type: ignore[arg-type]


@pytest.mark.core
def test_model_name_is_blip_base() -> None:
    assert captioner.MODEL_NAME == "Salesforce/blip-image-captioning-base"


# =============================================================================
# device()
# =============================================================================


@pytest.mark.core
def test_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    assert captioner.device() == "cpu"


# =============================================================================
# batch.py helpers
# =============================================================================


@pytest.mark.core
def test_find_images_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")  # excluded
    names = [p.name for p in batch.find_images(tmp_path)]
    assert names == ["a.png", "b.jpg"]


@pytest.mark.core
def test_caption_folder_skips_bad_files_and_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject fakes so no model is needed; a corrupt "image" must be skipped.
    monkeypatch.setattr(captioner, "_PROCESSOR", FakeProcessor())
    monkeypatch.setattr(captioner, "_MODEL", FakeModel())
    Image.new("RGB", (8, 8)).save(tmp_path / "good.png")
    (tmp_path / "bad.png").write_bytes(b"not really a png")
    out = tmp_path / "captions.txt"
    results = batch.caption_folder(tmp_path, out)
    assert [n for n, _ in results] == ["good.png"]
    assert "good.png: a fake caption" in out.read_text(encoding="utf-8")


@pytest.mark.core
def test_batch_loads_the_model_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []
    monkeypatch.setattr(
        captioner,
        "load_model",
        lambda name=captioner.MODEL_NAME: (
            loads.append(name),  # type: ignore[func-returns-value]  # tuple trick: discard None via [1:]
            FakeProcessor(),
            FakeModel(),
        )[1:],
    )
    for i in range(3):
        Image.new("RGB", (8, 8)).save(tmp_path / f"img{i}.png")
    rows = batch.caption_folder(tmp_path, tmp_path / "captions.txt")
    assert len(rows) == 3
    assert loads == [captioner.MODEL_NAME]


# =============================================================================
# Real BLIP integration - needs the `models` dependency group
# =============================================================================


@pytest.mark.models
def test_captions_a_real_image() -> None:
    img = Image.new("RGB", (64, 64), (30, 90, 220))
    text = captioner.caption(img)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


@pytest.mark.core
def test_load_model_reads_safetensors_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pinned commit must ship model.safetensors and the loader must insist on it:
    # otherwise transformers fetches the safetensors from an unpinned conversion PR and
    # downloads the weights twice (fresh-clone test, 2026-09-23).
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Fake:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> Fake:
            calls.append((cls.__name__, name, kwargs))
            return cls()

        def to(self, _device: str) -> Fake:
            return self

    fake = types.SimpleNamespace(
        BlipProcessor=type("BlipProcessor", (Fake,), {}),
        BlipForConditionalGeneration=type("BlipForConditionalGeneration", (Fake,), {}),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(captioner, "_PROCESSOR", None)
    monkeypatch.setattr(captioner, "_MODEL", None)
    monkeypatch.setattr(captioner, "device", lambda: "cpu")

    captioner.load_model()

    assert captioner.MODEL_REVISION == "4c26dfece70e02028433dd192458a54b390b85d2"
    assert calls[1] == (
        "BlipForConditionalGeneration",
        captioner.MODEL_NAME,
        {"revision": captioner.MODEL_REVISION, "use_safetensors": True},
    )
