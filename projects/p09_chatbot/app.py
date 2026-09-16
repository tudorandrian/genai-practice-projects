"""app.py — Flask chatbot web app over Blenderbot (Project P09).

Three layers in one app: a Hugging Face conversational model (``engine``), a
Flask JSON API that keeps conversation history, and an HTML/JS chat page.

Run:
    uv run python -m projects.p09_chatbot.app              # http://127.0.0.1:5000
    uv run python -m projects.p09_chatbot.app --host 0.0.0.0

Endpoints
    GET  /          -> the chat page (templates/index.html)
    POST /chatbot   {"message": "..."}  -> 200 {"reply": "..."} ; 400 {"error": ...}
    POST /reset     -> 200 {"status": "reset"} ; clears the conversation history

The generation call goes straight through ``engine.reply`` — tests
monkeypatch ``engine.reply`` so they never load the real model. This module
imports the ``engine`` *module*, not individual names from it, so a
monkeypatch of ``engine.reply`` (as opposed to a private copy bound by
``from engine import reply``) is visible here too.

Binds to ``127.0.0.1`` by default; pass ``--host 0.0.0.0`` to listen on every
interface.
"""

from __future__ import annotations

import argparse
import logging

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from projects.p09_chatbot import engine

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

app = Flask(__name__)
# Reject bodies above 64 KiB before Flask/Werkzeug buffers them for parsing —
# without this, a request can grow the process's memory (or, for a
# pathologically nested JSON body, blow the interpreter's recursion limit
# inside json.loads) before validation ever runs. Matches P07's cap so the
# two Flask APIs in this repository behave the same way here.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
# Allow calls from other origins. Fine for a local single-user demo; a real
# deployment would scope this to specific origins instead of the whole app.
CORS(app)

# Single shared conversation history. NOTE: one global conversation is fine
# for a single local user (dev server, one worker); a multi-user/production
# deployment must move to per-session state.
history: list[str] = []


@app.route("/")
def index() -> str:
    """Serve the chat page."""
    return render_template("index.html")


@app.route("/chatbot", methods=["POST"])
def chatbot() -> tuple[Response, int]:
    """Generate a reply for the JSON body's ``message``; 400 on invalid input."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "message" not in data:
        return jsonify({"error": "Send a JSON body with a 'message' field."}), 400
    message = data["message"]
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "The 'message' field must be a non-empty string."}), 400

    message = message.strip()
    answer = engine.reply(history, message)
    history.extend([message, answer])  # update history after the turn
    return jsonify({"reply": answer}), 200


@app.route("/reset", methods=["POST"])
def reset() -> tuple[Response, int]:
    """Clear the conversation history and start over."""
    history.clear()
    return jsonify({"status": "reset"}), 200


@app.errorhandler(400)
def _handle_400(_err: Exception) -> tuple[Response, int]:
    return jsonify({"error": "Bad request."}), 400


@app.errorhandler(404)
def _handle_404(_err: Exception) -> tuple[Response, int]:
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def _handle_405(_err: Exception) -> tuple[Response, int]:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the Blenderbot chat UI.")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST}).")
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the model once, then start the dev server."""
    args = parse_args(argv)
    engine.load_model()  # warm the cache before serving
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
