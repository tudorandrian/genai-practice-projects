# P10 - Meeting Assistant (Whisper + LLM)

## What it does

Chains two AI models from different domains into one pipeline: speech ->
text -> structured text. [Whisper](https://huggingface.co/openai/whisper-tiny.en)
(`openai/whisper-tiny.en`, via `transformers`) transcribes a meeting
recording, and the transcript is fed to an LLM behind one provider seam to
extract three fixed sections - topics discussed, decisions and action items.
This is the repository's first project where one model's output is the next
model's input, the composition pattern behind most real GenAI applications.

`assistant.py` is the whole pipeline in one module:

```
load_audio(path) -> np.ndarray                         # WAV -> float32 mono @ 16 kHz (scipy, no ffmpeg)
transcribe(audio_path) -> str                            # Whisper ASR, greedy decoding
build_prompt(transcript, section=None) -> str            # combined or single-section prompt
summarize_with_llm(prompt) -> str                         # THE provider seam
summarize_structured(transcript) -> str                   # one focused call per section, fixed headers
process_meeting(audio_path) -> {"transcript", "summary"}  # full chain, writes both files
```

`summarize_with_llm` dispatches on `MEETING_LLM_PROVIDER`: `ollama`
(default, a local model over HTTP, no key), `openai` (Chat Completions,
needs `OPENAI_API_KEY`), `local` (an offline `transformers` causal LM,
downloads its own weights), and `stub` (no model at all; it echoes the start
of the transcript - used by the tests and by `demo()`). If `ollama` is unreachable,
`summarize_with_llm` logs a warning and falls back to the stub with a
visible `[ollama unavailable ...]` marker instead of raising - see
[ADR 0007](../../docs/decisions/0007-p10-provider-seam-and-fallback.md).

`synthetic_audio.py` synthesizes two short, scripted meeting recordings with
`pyttsx3` so the pipeline can be exercised end to end with no copyrighted or
expiring demo download. A Gradio UI (`build_ui()`) is also in `assistant.py`.

## Run

```bash
uv sync --group models                                 # torch, transformers, gradio, pyttsx3
uv run p10-meeting-assistant --demo                     # Whisper transcript + stub summary, writes output/
uv run p10-meeting-assistant path/to/audio.wav          # transcribe + summarize one file
uv run p10-meeting-assistant                             # Gradio UI, http://127.0.0.1:7860
uv run p10-meeting-assistant --host 0.0.0.0              # UI: listen on every interface
uv run pytest projects/p10_meeting_assistant -q          # fast tests, no weights needed
```

`--verbose` logs at `INFO`; by default only the summary lines print.
`--demo` generates (or reuses) `data/standup.wav`, transcribes it with real
Whisper weights, summarizes it with the `stub` provider (force-pinned - see
"Design notes"), and writes `output/transcript.txt`, `output/summary.txt`
and `output/metrics.txt`.

To run the chain against a real LLM instead of the stub, set the provider
before the CLI mode (not `--demo`, which always pins `stub`):

```bash
docker compose --profile llm up -d        # Qwen2.5 1.5B served by Ollama on localhost
export MEETING_LLM_PROVIDER=ollama       # MEETING_OLLAMA_MODEL/_URL override the defaults
uv run p10-meeting-assistant path/to/audio.wav

# OpenAI
export MEETING_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...             # MEETING_OPENAI_MODEL overrides the default
uv run p10-meeting-assistant path/to/audio.wav
```

`.env.example` lists every key P10 reads. Nothing loads `.env` automatically:
export the variables, or copy the file to `.env` (gitignored) and run with
`uv run --env-file .env p10-meeting-assistant …`. No key is required for the default `ollama` provider (it degrades to the stub
if nothing is listening) or for `stub` itself.

## Example output

`uv run p10-meeting-assistant --demo`:

```
p10-meeting-assistant: ok
  provider: stub
  words: 73
  sections: 3
  wrote: output/
```

`seconds` varies run to run and is reported only on the console and in the
returned `DemoResult`, never in a committed file - running `--demo` twice in
a row leaves `git status` clean. `output/metrics.txt` (committed,
deterministic): `provider=stub`, `words=73`, `sections=3`, one `key=value`
per line.

`output/transcript.txt` (committed, deterministic - Whisper's transcription
of the synthesized standup recording, excerpt):

```
Good morning team. Today we discussed three topics. First, the mobile app
released timeline. Second, the customer feedback from last week. ...
```

`output/summary.txt` (committed, deterministic - excerpt):

```
## Topics discussed
- (stub) Good morning team. Today we discussed three topics. First, t

## Decisions
- (stub) Good morning team. Today we discussed three topics. First, t
```

The `stub` provider returns `"- (stub) "` plus the first 60 characters of
the transcript in its prompt, so every section repeats the same excerpt.
That proves Whisper's transcript reaches the summarizer and that the
three-section contract holds; it is not a summary. See "Design notes" for
why `demo()` pins it.

## Design notes

- **One provider seam, four backends.** `summarize_with_llm(prompt) -> str`
  is the only function `summarize_structured` calls into for text
  generation; swapping providers touches only its body, and no API key ever
  lives in code - every key/URL/model name is read from the environment at
  call time, not cached at import time, so tests and operators can override
  `MEETING_OLLAMA_URL`/`MEETING_OLLAMA_MODEL`/`MEETING_LLM_MODEL` without
  reloading the module.
- **`demo()` force-pins `stub`, not "whichever provider is configured".**
  The default provider is `ollama`; this machine (and most machines running
  the test suite) has no Ollama listening, and a machine that does would
  produce a different summary from Ollama's actual output. A committed
  artefact must not depend on which services happen to be running, so
  `demo()` sets `MEETING_LLM_PROVIDER=stub` for its own run, restoring
  whatever was set before it returns. Run the CLI directly (see "Run") to
  exercise a real provider.
- **Fixed section headers guarantee structure.** `summarize_structured`
  always assembles the summary under `## Topics discussed`, `## Decisions`
  and `## Action items`, one focused LLM call per section.
- **Visible, not silent, degradation.** Per
  [ADR 0007](../../docs/decisions/0007-p10-provider-seam-and-fallback.md),
  only the `ollama` branch falls back (to the stub, prefixed
  `[ollama unavailable - stub used]`) when the service is unreachable;
  `openai` and `local` still raise on failure, since a bad key or missing
  weights need the caller's attention.
- **Determinism.** `load_asr_model` passes `generate_kwargs={"num_beams": 1,
  "do_sample": False}` explicitly, so the same audio always transcribes to
  the same text. It also pins Whisper to `device="cpu"` and
  `dtype="float32"` for reproducibility rather than letting transformers
  pick an accelerator (on Apple Silicon it picked MPS, where Whisper
  transcribed healthy audio as noise; pinning CPU fixed that). The committed
  proofs come from Windows; on macOS the word and section counts
  (`words=73 sections=3`) matched, and the transcript text itself was not
  compared. The device and dtype actually used are logged on load.
  `synthetic_audio.generate()` also skips a script whose WAV
  already exists unless `force=True`, so repeated `--demo` calls reuse the
  same audio bytes. With the pinned `stub` provider, this keeps
  `output/transcript.txt` and `output/summary.txt` byte-identical across
  repeated `--demo` runs - WAV files are never tracked by git.
- **No ffmpeg needed for the common case.** `load_audio` decodes WAV with
  `scipy.io.wavfile` and resamples to 16 kHz with
  `scipy.signal.resample_poly`, entirely in-process. Non-WAV formats
  (mp3/m4a/...) fall back to Whisper's own file loader, which does need
  ffmpeg on `PATH`; `transcribe` reports that as a clean `ValueError`.
- **Output contract.** Library functions never `print`; they log through
  `logging.getLogger(__name__)`. `main()` prints at most six summary lines
  and sets the logging level (`WARNING`, or `INFO` with `--verbose`).

## Limits

- **`ollama`, `openai` and `local` each need their own setup** (`ollama
  serve` + a pulled model; an `OPENAI_API_KEY`; or a second Hugging Face
  download for `local`). The `core` tests force `stub` or a monkeypatched seam,
  and `demo()` force-pins `stub`; one `llm`-marked test runs the three-section
  summary against Qwen2.5 1.5B served by Ollama (`heavy.yml`'s `llm` job, or locally
  after `docker compose --profile llm up -d`). `openai` and `local` are never exercised.
  Measured with that model: all three sections are populated and name the right
  owners, but the 1.5B model repeats action items under "Decisions"; a larger model
  (`MEETING_OLLAMA_MODEL`) separates them better.
- **On macOS, `pyttsx3`'s driver (`NSSpeechSynthesizer`) writes AIFF/AIFF-C
  bytes to the `.wav` path it is given.** `synthetic_audio.generate()`
  detects this and converts uncompressed AIFF/AIFF-C output to a standard
  RIFF WAV in place immediately after synthesis, so `assistant.load_audio`
  (which only reads RIFF via `scipy.io.wavfile`) never sees a mislabelled
  file; a compressed or unrecognized file instead raises
  `TTSEngineUnavailableError`, degrading `demo()` to `skipped`. The macOS
  driver also writes asynchronously, so `generate()` waits (up to 60 s) until
  the file is long enough for its script and has stopped growing; a file
  that never gets there is deleted and the demo reports `skipped`.
- **On Linux, `pyttsx3` needs `espeak-ng`** (`sudo apt-get install
  espeak-ng`); without it, `--demo` reports `skipped` with that hint. Even
  with it installed, `pyttsx3` on GitHub's `ubuntu-latest` runner reported
  success without writing `standup.wav`, so the demo reports `skipped`
  there too, not `failed`. Whisper transcription of the synthesized audio is
  therefore verified on Windows and macOS only.
- **`whisper-tiny.en` is the smallest, fastest, English-only Whisper
  checkpoint** - a trade of accuracy for a fast CPU demo. A larger
  checkpoint (`openai/whisper-base.en`, `-small.en`, ...) would transcribe
  more accurately at proportionally more load/inference time; swapping it
  in only requires overriding `WHISPER_MODEL`.
- **The `stub` provider produces a summary with no real content** - it
  echoes the first 60 characters of the transcript, to prove the hand-off
  and the three-section contract deterministically, not to demonstrate
  summarization quality; see "Run" for exercising a real provider.
- **`local` downloads its own weights**, separate from Whisper's, and is
  meaningfully slower on CPU than a small model served by Ollama.
- **No authentication or rate limiting on the Gradio server.** `build_ui()`
  is a local demo UI, not hardened for public exposure; the CLI binds
  `127.0.0.1` by default - pass `--host 0.0.0.0` to listen everywhere.

## Datasets and licences

There is no external dataset. `synthetic_audio.py`'s `SCRIPTS` are two
hand-written, fixed meeting transcripts ("standup", "budget"), synthesized
into WAV files with `pyttsx3` - a wrapper around the OS's own text-to-speech
engine (SAPI5 on Windows, eSpeak on Linux), not a downloaded model. The WAV
files are never tracked by git (`.gitignore`:
`projects/*/data/*.wav`); `demo()` reuses them across runs instead of
committing them (see "Design notes").

The transcription model is
[`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en)
from the Hugging Face Hub (MIT-licensed by OpenAI), trained on large-scale
weakly supervised speech recognition data (Radford et al., *Robust Speech
Recognition via Large-Scale Weak Supervision*, 2022). Weights are downloaded
on first use and cached locally; this project only performs inference
against them. The default `local` LLM fallback, `Qwen/Qwen2.5-0.5B-Instruct`,
is Apache-2.0 licensed by Alibaba Cloud and likewise downloaded on first use
if that provider is selected.

## Courses drawn on

- 6 Building Generative AI-Powered Applications with Python
- 3 Generative AI: Prompt Engineering Basics
