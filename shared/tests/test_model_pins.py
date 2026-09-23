"""Every pinned Hugging Face revision must ship safetensors weights, so the loaders never
reach for another ref (fresh-clone test, 2026-09-23). Network: runs in heavy.yml."""

import pytest
import requests

from projects.p08_image_captioning import captioner
from projects.p09_chatbot import engine

PINS = [
    (captioner.MODEL_NAME, captioner.MODEL_REVISION),
    (engine.MODEL_NAME, engine.MODEL_REVISION),
]


@pytest.mark.network
@pytest.mark.parametrize(("repo", "revision"), PINS)
def test_pinned_revision_ships_safetensors(repo: str, revision: str) -> None:
    response = requests.get(
        f"https://huggingface.co/api/models/{repo}/revision/{revision}", timeout=30
    )
    response.raise_for_status()
    files = {sibling["rfilename"] for sibling in response.json()["siblings"]}
    assert "model.safetensors" in files
