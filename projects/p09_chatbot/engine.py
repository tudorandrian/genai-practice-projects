"""engine.py - Blenderbot conversational inference, decoupled from Flask.

Project P09. This is the model layer: it loads
``facebook/blenderbot-400M-distill`` once and turns a conversation history + a
new message into a reply. ``app.py`` imports from here so the model is loaded
a single time per process and the pure inference function ``generate_reply``
stays testable without the web server or the ~700 MB download.

The heavy imports (torch, transformers) are deferred into ``load_model`` so
this module - and the pure logic - can be imported and unit-tested with fakes
on a machine without those packages (only the ``models`` dependency group
needs them).

Run
    uv run p09-chatbot --demo             # three-turn demo, writes output/
    uv run pytest projects/p09_chatbot -q  # fast tests, no weights needed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"

MODEL_NAME = "facebook/blenderbot-400M-distill"
# The Hub commit the model loads from: the Hub's own `safetensors` conversion (refs/pr/7,
# by SFconvertbot), a direct child of eaaf64e that only adds model.safetensors with
# identical tensors. Pinning it (not eaaf64e, which has only pytorch_model.bin) keeps
# transformers from fetching weights from an unpinned ref and downloading them twice.
# Bump it deliberately.
MODEL_REVISION = "a5c7ef0e7e1109ef7b4af6f03841305d7d46fa59"
MAX_NEW_TOKENS = 60
HISTORY_WINDOW = 6  # keep the last N turns as context

DEMO_TURNS = (
    "Hello, how are you?",
    "What is your favorite hobby?",
    "That sounds fun, tell me more.",
)

# Two conversational turns used only for the curl-contract transcript below -
# kept separate from DEMO_TURNS so the conversation-content transcript and the
# HTTP-contract transcript each read as a self-contained, independent demo.
CURL_DEMO_TURNS = (
    "Hello! Who are you?",
    "What can you help me with?",
)

# Module-level singletons so the weights load exactly once per process.
_TOKENIZER: Any = None
_MODEL: Any = None


def load_model(model_name: str = MODEL_NAME) -> tuple[Any, Any]:
    """Load (once) and return the ``(tokenizer, model)`` pair.

    The first call downloads the weights from the Hugging Face Hub into the
    local cache; later calls reuse the in-process singletons. Imports
    transformers lazily so this module stays importable without it.
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        # Lazy heavy import so the module is importable/testable without transformers.
        from transformers import (  # lazy heavy import, see module docstring
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        revision = MODEL_REVISION if model_name == MODEL_NAME else "main"
        # The pinned default must load its own safetensors; another model keeps the
        # library default (None), which may fall back to pytorch_model.bin.
        use_safetensors = True if model_name == MODEL_NAME else None
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # Explicit `Any`: with transformers installed, `from_pretrained`'s overloads
        # would need a `# type: ignore` on some call sites; without transformers
        # installed (`ignore_missing_imports` makes the import `Any` already), that
        # ignore would be flagged as unused. Widening the type here avoids the
        # overload check in either environment, so no ignore comment is needed.
        model: Any = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision=revision, use_safetensors=use_safetensors
        )
        _MODEL = model
    return _TOKENIZER, _MODEL


def build_context(history: list[str], message: str) -> str:
    """Join the last ``HISTORY_WINDOW`` history entries with the new message.

    Pure and trivially testable: this is the fixed-window truncation that lets
    the bot answer follow-up questions coherently without unbounded context.
    """
    window = history[-HISTORY_WINDOW:]
    return "\n".join([*window, message])


def generate_reply(
    history: list[str],
    message: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a reply for ``message`` given the conversation ``history`` (pure).

    ``tokenizer``/``model`` are injected, keeping this free of global state and
    unit-testable with fakes. Raises ``TypeError`` if ``message`` is not a
    non-empty string.
    """
    if not isinstance(message, str) or not message.strip():
        raise TypeError("message must be a non-empty string")
    prompt_text = build_context(history, message.strip())
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    # Greedy decoding (num_beams=1, do_sample=False): Blenderbot's default is
    # 10-way beam search, ~10x slower on CPU. Greedy keeps each turn fast and,
    # set explicitly rather than relying on library defaults, is deterministic
    # across runs - required so output/conversation_transcript.txt is
    # byte-identical between consecutive `--demo` runs. max_length=None avoids
    # the "both max_new_tokens and max_length set" warning.
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        max_length=None,
    )
    return str(tokenizer.decode(output[0], skip_special_tokens=True)).strip()


def reply(history: list[str], message: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Convenience wrapper: generate a reply using the cached Blenderbot model."""
    tokenizer, model = load_model()
    return generate_reply(history, message, tokenizer, model, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# Demo, CLI
# ---------------------------------------------------------------------------


def _write_curl_demo(client: Any, app_module: Any) -> None:
    """Drive the ``/chatbot``, ``/reset`` and ``/`` HTTP contract through
    ``app.test_client()`` and write a deterministic curl-style transcript.

    No live server is started - every request goes straight through Flask's
    test client, exactly like ``demo()``'s conversation transcript above and
    ``projects/p07_sentiment_api/server.py``'s ``curl_demo.txt``, which this
    follows in format. Covers: a first turn, a follow-up turn that reuses the
    growing history, a missing-``message`` 400, ``/reset``, a wrong-method
    405 on ``/chatbot``, and ``GET /``.
    """
    app_module.history.clear()
    lines = [
        "# P09 chatbot - demo curl transcript",
        "# Generated via app.test_client() against /chatbot, /reset and / (no live server)",
        "",
    ]

    for turn in CURL_DEMO_TURNS:
        body = json.dumps({"message": turn})
        response = client.post("/chatbot", data=body, content_type="application/json")
        lines.append(
            "$ curl -s -X POST http://127.0.0.1:5000/chatbot "
            f"-H 'Content-Type: application/json' -d '{body}'"
        )
        lines.append(json.dumps(response.get_json(), sort_keys=True))
        lines.append("")

    missing_field = client.post("/chatbot", data="{}", content_type="application/json")
    lines.append(
        "$ curl -s -X POST http://127.0.0.1:5000/chatbot "
        "-H 'Content-Type: application/json' -d '{}'"
    )
    lines.append(json.dumps(missing_field.get_json(), sort_keys=True))
    lines.append("")

    reset_response = client.post("/reset")
    lines.append("$ curl -s -X POST http://127.0.0.1:5000/reset")
    lines.append(json.dumps(reset_response.get_json(), sort_keys=True))
    lines.append("")

    wrong_method = client.get("/chatbot")
    lines.append("$ curl -s -X GET http://127.0.0.1:5000/chatbot")
    lines.append(json.dumps(wrong_method.get_json(), sort_keys=True))
    lines.append("")

    lines.append("$ curl -s -o /dev/null -w 'GET / -> HTTP %{http_code}\\n' http://127.0.0.1:5000/")
    lines.append(f"GET / -> HTTP {client.get('/').status_code}")

    (OUT_DIR / "curl_demo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    app_module.history.clear()


def demo() -> DemoResult:
    """Run a three-turn scripted conversation through the real model.

    Tier ``models``: loads the real Blenderbot weights and never runs in CI.
    Posts ``DEMO_TURNS`` through the Flask test client (no live server),
    growing the conversation history one turn at a time, and writes a
    deterministic ``output/conversation_transcript.txt`` (conversation
    content) plus ``output/curl_demo.txt`` (the ``/chatbot``/``/reset``/``/``
    HTTP contract, via ``_write_curl_demo`` - also test-client-only) and
    ``output/metrics.txt``. Only the console summary and the returned
    ``DemoResult`` carry timing (``avg_s``); the three committed files never
    vary between runs, because generation is greedy and deterministic.
    """
    start = time.perf_counter()

    from projects.p09_chatbot import app as app_module

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app_module.history.clear()
    client = app_module.app.test_client()

    transcript = ["# P09 chatbot - demo conversation transcript", ""]
    for turn in DEMO_TURNS:
        response = client.post("/chatbot", json={"message": turn})
        answer = response.get_json()["reply"]
        transcript.append(f"user: {turn}")
        transcript.append(f"bot: {answer}")
        transcript.append("")
    if transcript and transcript[-1] == "":
        transcript.pop()
    (OUT_DIR / "conversation_transcript.txt").write_text(
        "\n".join(transcript) + "\n", encoding="utf-8"
    )

    _write_curl_demo(client, app_module)

    metrics_lines = [
        f"turns={len(DEMO_TURNS)}",
        f"model={MODEL_NAME}",
    ]
    (OUT_DIR / "metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    seconds = time.perf_counter() - start
    figures = {
        "turns": str(len(DEMO_TURNS)),
        "model": MODEL_NAME,
        "avg_s": f"{seconds / len(DEMO_TURNS):.2f}",
    }
    log.info("demo: %d turns with %s", len(DEMO_TURNS), MODEL_NAME)
    return DemoResult("p09-chatbot", "ok", figures, seconds=round(seconds, 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--demo", action="store_true", help="Run the demo (needs the models group).")
    p.add_argument("--verbose", action="store_true", help="Log at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p09-chatbot: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    print("Nothing to do: pass --demo, or run the app module directly to serve the chat UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
