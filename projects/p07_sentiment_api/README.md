# P07 — Flask Sentiment API

## What it does

A small Flask web service that scores short texts for sentiment and returns
the result as JSON. It is the first step from *scripts* to *services* in
this repository: the same kind of pure analysis function used by earlier
projects, now exposed over HTTP with a documented contract, JSON error
handling and a production-server option — instead of only being called
in-process.

The analysis engine is a hand-written lexicon (word -> integer weight) with
simple negation handling, so the whole thing runs offline: no model
download, no network call, no API key. The web layer never touches the
lexicon directly; `analyze_sentiment(text)` is a pure function that could be
swapped for a real model (e.g. `transformers`) without changing the HTTP
contract at all.

```
POST /sentiment {"text": "..."} -> analyze_sentiment(text) -> jsonify(...)
GET  /health                    -> {"status": "ok"}
```

`analyze_sentiment` tokenizes on whitespace, strips punctuation, lower-cases,
and sums `LEXICON` weights token by token, flipping the sign of the next
scored word after a negation token (`nu`, `fara`, `nici`, `niciun`, `nicio`)
so `"nu este bun"` (`"not good"`) scores negative rather than positive. The
result is `{"text", "score", "sentiment"}`, where `sentiment` is
`positive`/`negative`/`neutral` from the sign of `score`.

## Run

```bash
uv run p07-sentiment-api --demo                  # offline demo, writes output/
uv run p07-sentiment-api                         # dev server on http://127.0.0.1:5000
uv run p07-sentiment-api --host 0.0.0.0 --port 8080
uv run p07-sentiment-api --production            # served through waitress
uv run pytest projects/p07_sentiment_api -q      # tests, no server needed
```

With the dev server running:

```bash
curl -s -X POST http://127.0.0.1:5000/sentiment \
    -H 'Content-Type: application/json' \
    -d '{"text": "Un produs excelent si foarte util!"}'
# {"score": 3, "sentiment": "positive", "text": "Un produs excelent si foarte util!"}

curl -s http://127.0.0.1:5000/health
# {"status": "ok"}
```

`--verbose` logs at `INFO`; by default only the six-line demo summary
prints. `--demo` posts four fixed sentences through `app.test_client()`
(no server process involved) and writes `output/curl_demo.txt` (a
curl-transcript of the four request/response pairs) and `output/metrics.txt`
(lexicon size and request count) — both deterministic and committed.

### CLI flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--host HOST` | Bind address | `127.0.0.1` |
| `--port PORT` | Bind port | `5000` |
| `--production` | Serve through `waitress` instead of the Flask dev server | off |
| `--demo` | Run the offline demo | — |
| `--verbose` | Log at `INFO` | off |

The default host is `127.0.0.1`, never `0.0.0.0` — the dev server binds to
localhost only unless a caller explicitly opts into a wider bind address.

## Example output

`uv run p07-sentiment-api --demo`:

```
p07-sentiment-api: ok
  lexicon_size: 58
  requests: 4
  avg_ms: 7.10
  wrote: output/
```

`avg_ms` (mean request latency) varies run to run and is reported only on
the console and in the returned `DemoResult`, never in a committed file —
running `--demo` twice in a row leaves `git status` clean.

`output/metrics.txt` (committed, deterministic):

```
lexicon_size=58
requests=4
```

`output/curl_demo.txt` (committed, deterministic — excerpt):

```
$ curl -s -X POST http://127.0.0.1:5000/sentiment -H 'Content-Type: application/json' -d '{"text": "Un produs excelent si foarte util!"}'
{"score": 3, "sentiment": "positive", "text": "Un produs excelent si foarte util!"}
```

## Design notes

- **The analysis engine is decoupled from Flask.** `analyze_sentiment(text)`
  takes a string and returns a dict; it never imports `flask` or touches
  `request`/`jsonify`. The route handler is a thin adapter: validate the
  JSON body, call the pure function, wrap the result. This is what makes it
  possible to swap the lexicon for a real model later without touching the
  web layer or its tests.
- **English wire contract, Romanian lexicon data.** The JSON keys
  (`text`/`score`/`sentiment`/`error`) and labels
  (`positive`/`negative`/`neutral`) are English, matching every other
  project's identifier rule (see `docs/decisions/0006-p07-english-api-contract.md`).
  `LEXICON`'s words themselves stay Romanian — they are scoring data for a
  Romanian-text demo domain, not repository language, and are documented as
  such below.
- **Every error path returns JSON, never Flask's default HTML page.**
  A missing/absent `text` field, a non-string or blank `text`, an unparsable
  body, and the wrong HTTP method on `/sentiment` all return
  `{"error": "..."}` with an appropriate status code (400/405); an
  oversized body gets a JSON `413`, and the `500` handler covers anything
  else unexpected (e.g. a `RecursionError` from a deeply nested body) so it
  never falls through to Flask's default HTML error page either.
- **`--production` uses `waitress`, imported lazily.** The `core` dependency
  group (needed to just run the app) does not need `waitress`; the import
  happens only inside the `--production` branch of `main()`, so the plain
  dev-server path has one fewer dependency to install. `waitress` is listed
  in the `dev` group and is the documented way to run this service outside
  local development — the Flask dev server is explicitly not meant for
  production traffic.
- **Output contract.** Library functions (`analyze_sentiment`, the route
  handlers, `demo()`) never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines
  and sets the logging level (`WARNING`, or `INFO` with `--verbose`).
- **Determinism.** `demo()` always posts the same four sentences in the same
  order through `app.test_client()`, so `output/curl_demo.txt` and
  `output/metrics.txt` are byte-identical across runs; only the timing
  figure (`avg_ms`, in the console output and the returned `DemoResult`)
  varies.

## Limits

- **The lexicon is small and exact-match only.** 58 entries covering common
  positive/negative words and a handful of inflected forms; anything outside
  the list contributes zero weight, so subtler or misspelled sentiment is
  scored as neutral rather than approximated. Diacritics are folded before lookup
  (`excelentă`, `şi` and `și` match `excelenta`, `si`), so text written with or without
  them scores the same.
- **A negation stays active until the next scored word, however far away.**
  `"nu"`/`"fara"`/`"nici"`/`"niciun"`/`"nicio"` flip the sign of the next
  lexicon word regardless of how many unscored filler words — or sentence
  punctuation — sit in between (`"Nu stiu. Dar produsul este excelent!"`
  scores negative); nothing resets it at a full stop. A second negation
  before any scored word does not cancel the first (`"nu nu bun"` stays
  negative).
- **Request bodies over 64 KiB are rejected.** `MAX_CONTENT_LENGTH` caps the
  body Flask will buffer before parsing; a larger request gets a JSON `413`
  without ever reaching `analyze_sentiment`.
- **No authentication, rate limiting or CORS.** This is a demo service; it
  has no auth layer and is not hardened for public exposure.
- **`--production` still runs a single-process `waitress` server.** It is a
  real production-grade WSGI server (unlike the Flask dev server), but there
  is no process manager, TLS termination or horizontal scaling configured
  here — those are deployment concerns outside this project's scope.

## Datasets and licences

There is no external dataset. `LEXICON` is a small hand-written Romanian
sentiment word list, written for this repository specifically as demo data
(not derived from any corpus or third-party source) — kept in Romanian on
purpose, since the words themselves are the data being scored, not
repository identifiers or documentation language (see
`docs/decisions/0006-p07-english-api-contract.md`). The four demo sentences
in `demo()` are likewise hand-written for this project.

## Courses drawn on

- 5 Developing AI Applications with Python and Flask
