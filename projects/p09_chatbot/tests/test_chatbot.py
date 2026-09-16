"""Tests for engine.py and app.py (Project P09).

The fast tests inject a fake tokenizer/model (or monkeypatch ``engine.reply``),
so ``generate_reply`` and the Flask endpoints are exercised without
downloading Blenderbot or importing torch/transformers.

Run:
    uv run pytest projects/p09_chatbot -q
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from flask.testing import FlaskClient

from projects.p09_chatbot import app as app_module
from projects.p09_chatbot import engine

# No blanket module-level ``pytestmark`` here: every test is marked
# individually (``core``) so a future real-weights test added to this file
# stays opt-in under ``models`` rather than being swept up by a blanket
# marker - matching how projects/p08_image_captioning/tests/test_captioner.py
# handles the same real-weights-vs-fast-test split.


class FakeTokenizer:
    def __init__(self) -> None:
        self.last_text: str | None = None

    def __call__(
        self, text: str, return_tensors: str | None = None, truncation: bool | None = None
    ) -> dict[str, Any]:
        self.last_text = text
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, _ids: list[int], skip_special_tokens: bool = True) -> str:
        return "  fake reply  "  # padded to prove generate_reply strips it


class FakeModel:
    def __init__(self) -> None:
        self.max_new_tokens: int | None = None

    def generate(self, max_new_tokens: int | None = None, **_inputs: Any) -> list[list[int]]:
        self.max_new_tokens = max_new_tokens
        return [[0, 1, 2]]


@pytest.fixture
def client() -> Iterator[FlaskClient]:
    app_module.history.clear()
    with app_module.app.test_client() as test_client:
        yield test_client
    app_module.history.clear()


@pytest.fixture
def stub_reply(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str], str], str]:
    def fake(history: list[str], message: str) -> str:
        return f"ECHO:{message}"

    monkeypatch.setattr(engine, "reply", fake)
    return fake


# =============================================================================
# engine.build_context - pure
# =============================================================================


@pytest.mark.core
def test_build_context_uses_only_last_window() -> None:
    history = [f"m{i}" for i in range(10)]
    ctx = engine.build_context(history, "now")
    expected = "\n".join([*history[-engine.HISTORY_WINDOW :], "now"])
    assert ctx == expected


@pytest.mark.core
def test_context_keeps_only_the_last_six_turns() -> None:
    history = [f"turn {i}" for i in range(10)]
    ctx = engine.build_context(history, "now")
    assert "turn 3" not in ctx and "turn 4" in ctx and ctx.endswith("now")


# =============================================================================
# engine.generate_reply - pure, injected tokenizer/model
# =============================================================================


@pytest.mark.core
def test_generate_reply_returns_stripped() -> None:
    tok, model = FakeTokenizer(), FakeModel()
    out = engine.generate_reply(["hi", "hello"], "how are you?", tok, model)
    assert out == "fake reply"
    assert tok.last_text is not None and "how are you?" in tok.last_text
    assert model.max_new_tokens == engine.MAX_NEW_TOKENS


@pytest.mark.core
def test_generate_reply_rejects_empty() -> None:
    with pytest.raises(TypeError):
        engine.generate_reply([], "   ", FakeTokenizer(), FakeModel())


# =============================================================================
# /chatbot endpoint
# =============================================================================


@pytest.mark.core
def test_chat_endpoint_contract(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "reply", lambda history, message: f"STUB:{message}")
    resp = client.post("/chatbot", json={"message": "hello"})
    assert resp.status_code == 200 and set(resp.get_json()) == {"reply"}
    assert client.post("/chatbot", json={}).get_json() == {
        "error": "Send a JSON body with a 'message' field."
    }


@pytest.mark.core
def test_valid_message_200(
    client: FlaskClient, stub_reply: Callable[[list[str], str], str]
) -> None:
    r = client.post("/chatbot", json={"message": "Hi!"})
    assert r.status_code == 200
    assert r.get_json()["reply"] == "ECHO:Hi!"


@pytest.mark.core
def test_history_updates_after_turn(
    client: FlaskClient, stub_reply: Callable[[list[str], str], str]
) -> None:
    client.post("/chatbot", json={"message": "one"})
    client.post("/chatbot", json={"message": "two"})
    assert app_module.history == ["one", "ECHO:one", "two", "ECHO:two"]


@pytest.mark.core
def test_missing_message_400(client: FlaskClient) -> None:
    r = client.post("/chatbot", json={})
    assert r.status_code == 400
    assert "error" in r.get_json()


@pytest.mark.core
def test_empty_message_400(client: FlaskClient) -> None:
    r = client.post("/chatbot", json={"message": "   "})
    assert r.status_code == 400


@pytest.mark.core
def test_no_body_400(client: FlaskClient) -> None:
    r = client.post("/chatbot")
    assert r.status_code == 400
    assert "error" in r.get_json()


@pytest.mark.core
def test_wrong_method_405(client: FlaskClient) -> None:
    r = client.get("/chatbot")
    assert r.status_code == 405
    assert "error" in r.get_json()


@pytest.mark.core
def test_deeply_nested_json_returns_json_error(client: FlaskClient) -> None:
    # Small in bytes (well under MAX_CONTENT_LENGTH) but deep enough to blow
    # Python's recursion limit inside json.loads - request.get_json(silent=True)
    # only swallows ValueError/BadRequest, so the RecursionError used to
    # escape as Flask's default HTML 500 page. This never reaches ``engine``,
    # so it needs no model/weights and stays a fast ``core`` test.
    n = 3000
    body = "[" * n + "]" * n
    resp = client.post("/chatbot", data=body, content_type="application/json")
    assert resp.status_code != 200
    assert resp.content_type == "application/json"
    assert "error" in resp.get_json()


@pytest.mark.core
def test_oversized_body_returns_413_json(client: FlaskClient) -> None:
    body = json.dumps({"message": "a" * (app_module.app.config["MAX_CONTENT_LENGTH"] + 1)})
    resp = client.post("/chatbot", data=body, content_type="application/json")
    assert resp.status_code == 413
    assert resp.content_type == "application/json"
    assert "error" in resp.get_json()


@pytest.mark.core
def test_error_response_is_json(client: FlaskClient) -> None:
    r = client.post("/chatbot", json={})
    assert r.content_type == "application/json"


# =============================================================================
# /reset endpoint
# =============================================================================


@pytest.mark.core
def test_reset_clears_history(
    client: FlaskClient, stub_reply: Callable[[list[str], str], str]
) -> None:
    client.post("/chatbot", json={"message": "something"})
    assert app_module.history
    r = client.post("/reset")
    assert r.status_code == 200
    assert r.get_json() == {"status": "reset"}
    assert app_module.history == []


# =============================================================================
# / index route
# =============================================================================


@pytest.mark.core
def test_curl_demo_records_the_real_index_status(
    client: FlaskClient,
    stub_reply: Callable[[list[str], str], str],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """curl_demo.txt's `GET /` line must come from a real request, so a broken
    index route shows up in the proof instead of a hard-coded 200."""
    monkeypatch.setattr(engine, "OUT_DIR", tmp_path)

    class BrokenIndexClient:
        def post(self, *args: Any, **kwargs: Any) -> Any:
            return client.post(*args, **kwargs)

        def get(self, path: str) -> Any:
            if path == "/":
                return type("Response", (), {"status_code": 503})()
            return client.get(path)

    engine._write_curl_demo(client, app_module)
    assert (tmp_path / "curl_demo.txt").read_text(encoding="utf-8").endswith("GET / -> HTTP 200\n")

    engine._write_curl_demo(BrokenIndexClient(), app_module)
    assert (tmp_path / "curl_demo.txt").read_text(encoding="utf-8").endswith("GET / -> HTTP 503\n")


@pytest.mark.core
def test_index_serves_html(client: FlaskClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.content_type
    assert b"conversation" in r.data  # the chat container id
