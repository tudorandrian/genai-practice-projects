"""server.py — a small Flask sentiment-analysis API (Project P07).

The first step from *scripts* to *services* in this repository: the same kind
of pure Python analysis function used across earlier projects, now exposed
over HTTP as a JSON API. The analysis engine is a small local Romanian
lexicon, so the whole thing runs offline with a single ``pip install flask``
and no model download. The web layer never touches the lexicon directly —
swap ``analyze_sentiment`` for a real model (e.g. ``transformers``) and the
HTTP contract is unchanged.

Endpoints
    POST /sentiment   {"text": "..."}  -> 200 {"text", "score", "sentiment"}
                                          400 {"error": "..."} on bad input
    GET  /health                        -> 200 {"status": "ok"}

Run
    uv run p07-sentiment-api                    # dev server, http://127.0.0.1:5000
    uv run p07-sentiment-api --production       # waitress, production-grade WSGI
    uv run p07-sentiment-api --demo             # offline demo, writes output/
    uv run pytest projects/p07_sentiment_api -q # tests, no server needed
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import time
import unicodedata
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"

# ---------------------------------------------------------------------------
# Analysis engine — a pure function, fully decoupled from Flask
# ---------------------------------------------------------------------------

# Sentiment lexicon: word -> weight (+/-1 mild, +/-2 strong). The words are
# Romanian — this is data for the demo domain, not repository language, so it
# is kept as-is (see the README's "Datasets and licences" section). Common
# feminine/plural forms are listed explicitly (exact match only, so no false
# hits like "bun" wrongly matching "bunic").
LEXICON: dict[str, int] = {
    # positive
    "excelent": 2,
    "excelenta": 2,
    "minunat": 2,
    "minunata": 2,
    "extraordinar": 2,
    "perfect": 2,
    "perfecta": 2,
    "superb": 2,
    "superba": 2,
    "bun": 1,
    "buna": 1,
    "buni": 1,
    "bune": 1,
    "util": 1,
    "utila": 1,
    "utile": 1,
    "multumit": 1,
    "multumita": 1,
    "placut": 1,
    "placuta": 1,
    "rapid": 1,
    "rapida": 1,
    "recomand": 1,
    "frumos": 1,
    "frumoasa": 1,
    "ieftin": 1,
    "ieftina": 1,
    "prietenos": 1,
    "fericit": 1,
    "fericita": 1,
    # negative
    "prost": -2,
    "proasta": -2,
    "groaznic": -2,
    "groaznica": -2,
    "oribil": -2,
    "oribila": -2,
    "dezastru": -2,
    "dezastruos": -2,
    "scump": -1,
    "scumpa": -1,
    "slab": -1,
    "slaba": -1,
    "dezamagit": -1,
    "dezamagita": -1,
    "inutil": -1,
    "inutila": -1,
    "lent": -1,
    "lenta": -1,
    "defect": -1,
    "defecta": -1,
    "trist": -1,
    "trista": -1,
    "nemultumit": -1,
    "nemultumita": -1,
    "urat": -1,
    "urata": -1,
    "rau": -1,
    "rea": -1,
}

# Words that flip the polarity of the next sentiment-bearing word ("nu bun").
NEGATIONS: frozenset[str] = frozenset({"nu", "fara", "nici", "niciun", "nicio"})

_PUNCT = string.punctuation + "„”“…–—"


def _fold_diacritics(token: str) -> str:
    """``"excelentă"`` -> ``"excelenta"``: decompose, then drop the combining marks.

    The lexicon lists plain forms, and Romanian is normally written with ă, â, î, ș, ț
    (or the older cedilla ş, ţ); without this, correctly spelled text scored neutral.
    """
    decomposed = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    """Lower-case each whitespace token, strip surrounding punctuation, fold diacritics."""
    return [_fold_diacritics(token.strip(_PUNCT).lower()) for token in text.split()]


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Score ``text`` and label it positive / negative / neutral.

    Pure and deterministic: sums lexicon weights, flipping the sign of the
    next scored word after a negation token ("nu este bun" -> negative).
    Returns a dict with the original ``text``, an integer ``score`` and the
    ``sentiment`` label. Raises ``TypeError`` if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    score = 0
    negation_active = False
    for token in _tokenize(text):
        if not token:
            continue
        if token in NEGATIONS:
            negation_active = True
            continue
        weight = LEXICON.get(token, 0)
        if weight != 0:
            score += -weight if negation_active else weight
            negation_active = False

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"text": text, "score": score, "sentiment": label}


# ---------------------------------------------------------------------------
# Web layer — Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)
# Reject bodies above 64 KiB before Flask/Werkzeug buffers them for parsing —
# without this, a request can grow the process's memory (or, for a
# pathologically nested JSON body, blow the interpreter's recursion limit
# inside json.loads) before validation ever runs.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


@app.route("/sentiment", methods=["POST"])
def sentiment() -> tuple[Response, int]:
    """Analyse the JSON body's ``text`` field; 400 on any invalid input."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Send a JSON body with a 'text' field."}), 400
    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "The 'text' field must be a non-empty string."}), 400
    return jsonify(analyze_sentiment(text)), 200


@app.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """Liveness probe for monitoring/orchestration."""
    return jsonify({"status": "ok"}), 200


@app.errorhandler(400)
def _handle_400(_err: Exception) -> tuple[Response, int]:
    return jsonify({"error": "Bad request."}), 400


@app.errorhandler(404)
def _handle_404(_err: Exception) -> tuple[Response, int]:
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def _handle_405(_err: Exception) -> tuple[Response, int]:
    """Return JSON (not the default HTML page) for a wrong HTTP method."""
    return jsonify({"error": "HTTP method not allowed for this route."}), 405


@app.errorhandler(413)
def _handle_413(_err: Exception) -> tuple[Response, int]:
    """Return JSON for a body over MAX_CONTENT_LENGTH."""
    return jsonify({"error": "Request body too large."}), 413


@app.errorhandler(500)
def _handle_500(err: Exception) -> tuple[Response, int]:
    """Return JSON for anything unexpected below (e.g. a RecursionError from
    parsing a deeply nested — but under the size cap — JSON body), instead of
    Flask's default HTML error page. The exception is still logged."""
    log.error("unhandled error: %s", err, exc_info=err)
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Demo, CLI
# ---------------------------------------------------------------------------

DEMO_SENTENCES = (
    "Un produs excelent si foarte util!",
    "Serviciu groaznic si lent",
    "Nu este bun",
    "Cartea este pe raft",
)


def demo() -> DemoResult:
    """Post four sentences through ``app.test_client()`` (no live server) and
    write a deterministic curl-style transcript plus a deterministic metrics
    file. Timing is reported only in the console summary and the returned
    ``DemoResult`` — the two committed files never vary between runs."""
    start = time.perf_counter()
    client = app.test_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transcript = [
        "# P07 sentiment API — demo curl transcript",
        "# Generated via app.test_client() against POST /sentiment (no live server)",
        "",
    ]
    for text in DEMO_SENTENCES:
        body = json.dumps({"text": text})
        response = client.post("/sentiment", data=body, content_type="application/json")
        payload = response.get_json()
        transcript.append(
            "$ curl -s -X POST http://127.0.0.1:5000/sentiment "
            f"-H 'Content-Type: application/json' -d '{body}'"
        )
        transcript.append(json.dumps(payload, sort_keys=True))
        transcript.append("")
    if transcript and transcript[-1] == "":
        transcript.pop()  # no trailing blank line before the final newline
    (OUT_DIR / "curl_demo.txt").write_text("\n".join(transcript) + "\n", encoding="utf-8")

    metrics_lines = [
        f"lexicon_size={len(LEXICON)}",
        f"requests={len(DEMO_SENTENCES)}",
    ]
    (OUT_DIR / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    seconds = time.perf_counter() - start
    figures = {
        "lexicon_size": str(len(LEXICON)),
        "requests": str(len(DEMO_SENTENCES)),
        "avg_ms": f"{(seconds / len(DEMO_SENTENCES)) * 1000:.2f}",
    }
    log.info("demo: %d requests, lexicon_size=%d", len(DEMO_SENTENCES), len(LEXICON))
    return DemoResult("p07-sentiment-api", "ok", figures, seconds=round(seconds, 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000).")
    p.add_argument(
        "--production",
        action="store_true",
        help="Serve through waitress (a production-grade WSGI server) instead of the dev server.",
    )
    p.add_argument("--demo", action="store_true", help="Run the offline demo.")
    p.add_argument("--verbose", action="store_true", help="Log at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p07-sentiment-api: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.production:
        import waitress

        waitress.serve(app, host=args.host, port=args.port)
        return 0

    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
