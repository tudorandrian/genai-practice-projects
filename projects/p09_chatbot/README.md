# P09 - Flask Chatbot over a Hugging Face Model

## What it does

A local, browser-based chatbot: a Flask JSON API in front of
[Blenderbot](https://huggingface.co/facebook/blenderbot-400M-distill)
(`facebook/blenderbot-400M-distill`), plus a small HTML/JS chat page served
by the same app. This is the second `models`-tier project in the
repository - like P08, it loads real Hugging Face weights - but the shape
here is a conversational, stateful web service rather than a stateless
single-shot inference call.

`engine.py` is the model layer: `load_model()` downloads Blenderbot once
into the Hugging Face cache and keeps it in module-level singletons,
`build_context(history, message)` joins the last `HISTORY_WINDOW` (6) turns
of conversation with the new message into one prompt, and
`generate_reply(history, message, tokenizer, model)` is a pure function that
takes an injected tokenizer/model pair so it is unit-testable with fakes,
without downloading anything. `reply(history, message)` is the convenience
wrapper that loads the cached model and calls `generate_reply`.

```
build_context(history, message) -> str                      # pure, fixed-window prompt
generate_reply(history, message, tokenizer, model) -> str    # pure, injectable, testable
reply(history, message) -> str                               # wrapper: load_model() + generate_reply()
```

`app.py` is a thin Flask layer on top of `engine`: it keeps one shared
`history: list[str]`, exposes `/chatbot` and `/reset`, and serves
`templates/index.html` at `/`. It never touches `transformers` or `torch`
directly and imports the `engine` *module* (not individual names from it),
so tests that monkeypatch `engine.reply` are honoured here too - a
`from engine import reply` would have bound a private copy of the function
that a later patch could not reach.

## Run

```bash
uv sync --group models                              # torch, transformers
uv run p09-chatbot --demo                            # three-turn demo, writes output/
uv run python -m projects.p09_chatbot.app            # chat UI, http://127.0.0.1:5000
uv run python -m projects.p09_chatbot.app --host 0.0.0.0   # listen on every interface
uv run pytest projects/p09_chatbot -q                # fast tests, no weights needed
```

With the dev server running:

```bash
curl -s -X POST http://127.0.0.1:5000/chatbot \
    -H 'Content-Type: application/json' \
    -d '{"message": "Hello, how are you?"}'
# {"reply": "..."}

curl -s -X POST http://127.0.0.1:5000/reset
# {"status": "reset"}
```

`--verbose` logs at `INFO`; by default only the four-line demo summary
prints. `--demo` posts three fixed messages through `app.test_client()` (no
live server - see "Never start the server" below) and writes
`output/conversation_transcript.txt` (the three user/bot turns),
`output/curl_demo.txt` (curl-style requests and responses for `/chatbot`,
`/reset` and `/`, also through the test client) and `output/metrics.txt`
(turn count and model name) - all deterministic and committed.

### Endpoints

| Route | Method | Body | Success | Error |
|---|---|---|---|---|
| `/` | GET | - | chat page (HTML) | - |
| `/chatbot` | POST | `{"message": "..."}` | 200 `{"reply": "..."}` | 400 `{"error": "..."}` |
| `/reset` | POST | - | 200 `{"status": "reset"}` | - |

## Example output

`uv run p09-chatbot --demo`:

```
p09-chatbot: ok
  turns: 3
  model: facebook/blenderbot-400M-distill
  avg_s: 1.84
  wrote: output/
```

`avg_s`/`seconds` (generation time) vary run to run and are reported only on
the console and in the returned `DemoResult`, never in a committed file -
running `--demo` twice in a row leaves `git status` clean, because
generation is greedy and deterministic (`num_beams=1, do_sample=False`, set
explicitly rather than relying on Blenderbot's beam-search default).

`output/metrics.txt` (committed, deterministic):

```
turns=3
model=facebook/blenderbot-400M-distill
```

`output/conversation_transcript.txt` (committed, deterministic - excerpt):

```
# P09 chatbot - demo conversation transcript

user: Hello, how are you?
bot: ...
```

## Design notes

- **Engine/web separation.** `engine.py` never imports `flask`;
  `generate_reply` is a pure `(history, message, tokenizer, model) -> str`
  function that never touches Flask's request/response objects. `app.py`
  validates the JSON body, calls `engine.reply`, and updates `history` -
  the same split used by P07 (analysis function vs. route handler) and P08
  (inference function vs. UI).
- **Lazy heavy imports.** `transformers` is imported inside `load_model()`,
  not at module level, so `engine.py` is importable - and its pure logic
  testable - without the `models` dependency group installed; `core` tests
  monkeypatch a fake tokenizer/model or `engine.reply` instead of loading
  real weights, per the fixed-window and endpoint-contract tests.
- **Fixed-window context.** `build_context` keeps only the last
  `HISTORY_WINDOW` (6) history entries plus the new message, so the prompt
  passed to Blenderbot stays bounded regardless of how long the
  conversation runs; this is what `test_context_keeps_only_the_last_six_turns`
  proves directly.
- **Deterministic decoding.** `generate_reply` passes `num_beams=1,
  do_sample=False` explicitly rather than relying on Blenderbot's default
  (10-way beam search, which is both non-deterministic to rely on across
  library versions and roughly 10x slower on CPU). That is what keeps
  `output/conversation_transcript.txt` byte-identical across repeated
  `--demo` runs.
- **Never start the server for tests or the demo.** Both the test suite and
  `demo()` drive the app through `app.test_client()`, never `app.run()` - a
  blocking dev server has no place in an automated run. `app.py`'s own
  `main()` (used for interactive local use only) still starts the real dev
  server, bound to `127.0.0.1` by default; pass `--host 0.0.0.0` to opt into
  listening on every interface.
- **Output contract.** `engine.py` and `app.py` never `print`; they log
  through `logging.getLogger(__name__)`. `engine.py`'s `main()` prints at
  most four summary lines and sets the logging level (`WARNING`, or `INFO`
  with `--verbose`).
- **Single shared history.** `app.history` is one global conversation list,
  which is fine for a single local user hitting one dev-server worker; a
  multi-user or production deployment would need per-session state (e.g. a
  session id in the request, history keyed by it) instead.

## Limits

- **Blenderbot-400M-distill is a 2020 model, kept for its CPU footprint.**
  It runs a full generation turn on CPU in a few seconds with no GPU and no
  quantization, which is why it is still the default here for a
  reproducible, offline-friendly demo. A closer-to-current alternative is
  `Qwen/Qwen2.5-0.5B-Instruct`, a small 2026-era instruction-tuned model;
  swapping it in only requires overriding `MODEL_NAME` (`engine.load_model`
  takes the model name as a parameter) - no other code changes, though
  prompt formatting and generation quality would differ from Blenderbot's.
- **`CORS(app)` is applied app-wide.** That is fine for a local, single-user
  demo reached only from `http://127.0.0.1`, but it would not be an
  appropriate setting for a real deployment, which should scope allowed
  origins explicitly instead of allowing every origin.
- **No authentication, rate limiting, or per-session isolation.** `app.py`
  is a local demo UI, not hardened for public exposure; see "Single shared
  history" above.
- **Replies are short and sometimes generic.** Blenderbot-400M-distill is a
  small distilled model; replies are capped at `MAX_NEW_TOKENS=60` and can
  be repetitive or only loosely on-topic, especially past the 6-turn
  context window.

## Datasets and licences

There is no external dataset. `DEMO_TURNS` in `engine.demo()` are three
hand-written, fixed conversational prompts used to exercise the model
end-to-end; they are not sourced from any corpus.

The model is
[`facebook/blenderbot-400M-distill`](https://huggingface.co/facebook/blenderbot-400M-distill)
from the Hugging Face Hub, licensed Apache-2.0, distilled from the larger
BlenderBot 1.0 conversational model (Roller et al., *Recipes for Building an
Open-Domain Chatbot*, 2020) and pretrained/fine-tuned on public
conversational datasets by the model's authors. Weights are downloaded on
first use and cached locally; this project only performs inference against
them.

## Courses drawn on

- 6 Building Generative AI-Powered Applications with Python
