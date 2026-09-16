"""Tests for server.py: the pure analysis function, the Flask endpoints, error
handling, and the CLI (dev server vs. waitress production flag).

Run:
    uv run pytest projects/p07_sentiment_api -q
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from flask.testing import FlaskClient

from projects.p07_sentiment_api import server

pytestmark = pytest.mark.core


@pytest.fixture
def client() -> Iterator[FlaskClient]:
    with server.app.test_client() as test_client:
        yield test_client


# =============================================================================
# Pure function
# =============================================================================


def test_positive() -> None:
    result = server.analyze_sentiment("Un produs excelent si foarte util")
    assert result["sentiment"] == "positive"
    assert result["score"] > 0
    assert isinstance(result["score"], int)


def test_negative() -> None:
    result = server.analyze_sentiment("Serviciu groaznic si foarte lent")
    assert result["sentiment"] == "negative"
    assert result["score"] < 0


def test_neutral() -> None:
    result = server.analyze_sentiment("Merele sunt pe masa din bucatarie")
    assert result["sentiment"] == "neutral"
    assert result["score"] == 0


def test_negation_flip() -> None:
    # "nu ... bun" must flip a positive word to negative.
    assert server.analyze_sentiment("Nu este bun")["sentiment"] == "negative"
    assert server.analyze_sentiment("Nu este prost")["sentiment"] == "positive"


def test_text_with_romanian_diacritics_matches_the_lexicon() -> None:
    # Romanian is normally written with diacritics; the lexicon lists the plain forms.
    # Both comma-below (ș, ț) and the older cedilla (ş, ţ) spellings must match.
    positive = server.analyze_sentiment("Mâncarea a fost excelentă și plăcută")
    assert (positive["score"], positive["sentiment"]) == (3, "positive")
    assert server.analyze_sentiment("Aplicaţia e lentă şi scumpă")["score"] == -2
    assert server.analyze_sentiment("Nu este bună")["sentiment"] == "negative"
    assert server.analyze_sentiment("Fără defecte, rapidă")["text"] == "Fără defecte, rapidă"


def test_negation_carries_across_sentence_punctuation() -> None:
    # Pins the documented Limit: a negation stays active until the next
    # scored word, even across a full stop and an intervening filler
    # sentence — it is not reset at punctuation.
    result = server.analyze_sentiment("Nu stiu. Dar produsul este excelent!")
    assert result["sentiment"] == "negative"
    assert result["score"] < 0


def test_type_error_on_non_string() -> None:
    with pytest.raises(TypeError):
        server.analyze_sentiment(123)  # type: ignore[arg-type]


# =============================================================================
# /sentiment endpoint
# =============================================================================


def test_positive_200(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={"text": "Un produs excelent si foarte util!"})
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"text", "score", "sentiment"}
    assert body["sentiment"] == "positive"
    assert isinstance(body["score"], int)


def test_negative_200(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={"text": "Experienta groaznica"})
    assert r.status_code == 200
    assert r.get_json()["sentiment"] == "negative"


def test_neutral_200(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={"text": "Cartea este pe raft"})
    assert r.status_code == 200
    assert r.get_json()["sentiment"] == "neutral"


def test_missing_text_400(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_empty_text_400(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={"text": "   "})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_no_body_400(client: FlaskClient) -> None:
    r = client.post("/sentiment")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_non_string_text_400(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={"text": 42})
    assert r.status_code == 400


def test_error_response_is_json(client: FlaskClient) -> None:
    r = client.post("/sentiment", json={})
    assert r.content_type == "application/json"  # not an HTML page


def test_wrong_method_405(client: FlaskClient) -> None:
    r = client.get("/sentiment")
    assert r.status_code == 405
    assert "error" in r.get_json()


# =============================================================================
# /health endpoint
# =============================================================================


def test_health_ok(client: FlaskClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


# =============================================================================
# New tests (Task 9): English error contract + waitress production flag
# =============================================================================


def test_invalid_json_body_returns_english_error(client: FlaskClient) -> None:
    resp = client.post("/sentiment", data="not json", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Send a JSON body with a 'text' field."}


def test_contract_keys_are_english() -> None:
    result = server.analyze_sentiment("Un produs excelent")
    assert set(result) == {"text", "score", "sentiment"}
    assert result["sentiment"] == "positive"


def test_deeply_nested_json_returns_json_error(client: FlaskClient) -> None:
    # Small in bytes (well under MAX_CONTENT_LENGTH) but deep enough to blow
    # Python's recursion limit inside json.loads — request.get_json(silent=True)
    # only swallows ValueError/BadRequest, so the RecursionError used to
    # escape as Flask's default HTML 500 page.
    n = 3000
    body = "[" * n + "]" * n
    resp = client.post("/sentiment", data=body, content_type="application/json")
    assert resp.status_code != 200
    assert resp.content_type == "application/json"
    assert "error" in resp.get_json()


def test_oversized_body_returns_413_json(client: FlaskClient) -> None:
    body = json.dumps({"text": "a" * (server.app.config["MAX_CONTENT_LENGTH"] + 1)})
    resp = client.post("/sentiment", data=body, content_type="application/json")
    assert resp.status_code == 413
    assert resp.content_type == "application/json"
    assert "error" in resp.get_json()


def test_production_flag_uses_waitress(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(
        "waitress.serve", lambda app, host, port: called.update(host=host, port=port)
    )
    assert server.main(["--production", "--port", "8123"]) == 0
    assert called == {"host": "127.0.0.1", "port": 8123}
