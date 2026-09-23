"""assistant.py - chain OpenAI Whisper (STT) into an LLM summarizer.

Project P10. This is the first project in the repository that chains two AI
models from different domains: speech -> language. Audio is transcribed with
Whisper, and the transcript is fed to an LLM with a structured prompt that
extracts three sections - topics, decisions and action items. The output of
the first model is the input of the second: the fundamental pattern behind
composed GenAI applications.

Design notes
  * The LLM provider is isolated behind one seam, ``summarize_with_llm(prompt)
    -> str``, dispatched by the ``MEETING_LLM_PROVIDER`` env var (default
    ``ollama`` = a local model served by Ollama; ``openai``, a ``local``
    transformers fallback, and ``stub`` are also available). No API key ever
    lives in the code - keys are read from the environment. The fully-local
    chain (Whisper + Ollama) runs with no key at all, and keeps the LLM
    weights outside this repo's Hugging Face cache. If Ollama is unreachable,
    ``summarize_with_llm`` logs a warning and falls back to the stub with a
    visible ``[ollama unavailable ...]`` marker rather than raising.
  * Audio is decoded with scipy (WAV) and resampled to 16 kHz in-process, so
    the common case needs no ffmpeg. Other containers (mp3) fall back to
    Whisper's own file loader, which does need ffmpeg - reported as a clean
    error if it is absent.

Run
    uv run p10-meeting-assistant path/to/audio.wav   # CLI: transcribe + summarize
    uv run p10-meeting-assistant --demo               # offline-ish demo, writes output/
    uv run p10-meeting-assistant                       # Gradio UI on 127.0.0.1:7860
    uv run pytest projects/p10_meeting_assistant -q    # fast tests, no weights needed

Dependencies  transformers, torch, gradio, scipy, numpy - the ``models`` group.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
DATA_DIR = HERE / "data"

WHISPER_MODEL = "openai/whisper-tiny.en"
# The Hub commit the committed proofs were produced with. Pinned so that a re-published
# model cannot silently change results or the code that loads it; bump it deliberately.
WHISPER_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
# LLM: default provider is Ollama, so the model lives outside this repo's
# Python cache. The transformers "local" provider is kept as an opt-in
# offline fallback. Every knob is read from the environment at call time
# (not cached at import time) so tests - and operators - can override any of
# them without reloading the module.
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LOCAL_LLM_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # Hub commit, as above
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
TARGET_SR = 16000  # Whisper expects 16 kHz mono
CHUNK_LENGTH_S = 30  # segment long audio
LLM_MAX_NEW_TOKENS = 160  # per focused section call

# Audio that parses and transcribes without raising can still be broken (a
# truncated file, or Whisper emitting noise). If the transcript's word count
# falls below this fraction of the synthesized script's own word count,
# `demo()` reports `failed`, not `skipped`: the cause may be this project's
# own code, so it must not be waved through as an environment limitation.
MIN_TRANSCRIPT_WORD_RATIO = 0.5

# The three sections the summary must always contain. Assembling them under
# fixed headers guarantees every section exists regardless of model quality.
SECTIONS: list[tuple[str, str]] = [
    (
        "Topics discussed",
        "List up to five main topics discussed in the meeting below. "
        "Answer as a short bulleted list, one topic per line starting with '- '.",
    ),
    (
        "Decisions",
        "List every decision that was made in the meeting below. "
        "Answer as a short bulleted list, one decision per line starting with '- '.",
    ),
    (
        "Action items",
        "List every action item in the meeting below, including who is responsible "
        "and the deadline. Answer as a bulleted list, one item per line with '- '.",
    ),
]

# Whisper runs on CPU in float32, pinned explicitly (see load_asr_model).
# ASR_DTYPE is passed as a string so this module (and its core-tier tests)
# never needs torch just to name the dtype.
ASR_DEVICE = "cpu"
ASR_DTYPE = "float32"

_ASR: Any = None
# One-line runtime facts (device/dtype the model actually landed on, torch
# and transformers versions), captured when the pipeline is first loaded so
# demo()'s failure note can carry them next to the audio facts.
_ASR_FACTS: str | None = None
_LOCAL_LLM: Any = None
# A dedicated child logger for the ASR runtime-facts line, so it can be made
# visible under `uv run demo --all` (see load_asr_model) without also
# un-silencing every other INFO line this module logs.
asr_log = logging.getLogger(f"{__name__}.asr")


# =============================================================================
# Step 1 - Speech to text (Whisper)
# =============================================================================


def load_audio(path: str | Path) -> np.ndarray:
    """Decode a WAV file into a float32 mono waveform resampled to 16 kHz.

    Uses scipy only (no ffmpeg). Raises ``ValueError`` on a non-WAV / unreadable
    file so the caller can show a friendly error. Integer PCM is scaled to
    [-1, 1]; 8-bit WAV is unsigned (silence is 128), so it is centred first.
    """
    path = Path(path)
    try:
        sample_rate, data = wavfile.read(path)
    except Exception as exc:  # not a wav, corrupt, missing
        raise ValueError(f"Cannot read '{path.name}' as a WAV file: {exc}") from exc

    data = np.asarray(data)
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    # normalise integer PCM to float32 in [-1, 1]
    if data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if sample_rate != TARGET_SR:  # resample to 16 kHz
        data = resample_poly(data, TARGET_SR, sample_rate).astype(np.float32)
    return data


def load_asr_model() -> Any:
    """Load (once) and return the Whisper ASR pipeline.

    Decoding is set explicitly to greedy (``num_beams=1, do_sample=False``)
    rather than relying on the pipeline's defaults, so a given audio input
    always transcribes to the same text - required for
    ``output/transcript.txt`` to be byte-identical across repeated ``--demo``
    runs. Imports transformers lazily so this module stays importable without
    it (only the ``models`` dependency group needs it).

    The device and dtype are pinned (``ASR_DEVICE``/``ASR_DTYPE``). Once
    loaded, the device/dtype the model *actually* landed on plus the
    torch/transformers versions are logged as one INFO line (see
    ``asr_runtime_facts``) and kept for ``demo()``'s failure note.
    """
    global _ASR, _ASR_FACTS
    if _ASR is None:
        from transformers import pipeline  # lazy heavy import, see module docstring

        from projects.p10_meeting_assistant import synthetic_audio

        # Pinned to CPU/float32 rather than left to transformers' device and
        # dtype auto-selection: (1) the committed proofs were produced on CPU,
        # and different backends/dtypes give different floating-point results;
        # (2) on Apple Silicon, auto-selection placed Whisper on the MPS
        # backend, where it transcribed healthy audio as repeated-token noise,
        # and pinning CPU fixed it.
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            revision=WHISPER_REVISION,
            chunk_length_s=CHUNK_LENGTH_S,
            generate_kwargs={"num_beams": 1, "do_sample": False},
            device=ASR_DEVICE,
            dtype=ASR_DTYPE,
        )
        _ASR_FACTS = asr_runtime_facts(_ASR)
        synthetic_audio._ensure_facts_logging_visible(asr_log)
        asr_log.info(_ASR_FACTS)
    return _ASR


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def asr_runtime_facts(asr: Any) -> str:
    """One single-line summary of where a loaded ASR pipeline actually runs:
    the device and dtype read back from its model (not the arguments passed
    to ``pipeline()``), plus the installed torch and transformers versions.
    Best-effort: an attribute that cannot be read is reported as
    ``unknown`` rather than raising."""
    model = getattr(asr, "model", None)
    device = getattr(model, "device", None)
    if device is None:
        device = getattr(asr, "device", None)
    dtype = getattr(model, "dtype", None)
    return (
        f"whisper: model={WHISPER_MODEL} device={device if device is not None else 'unknown'} "
        f"dtype={dtype if dtype is not None else 'unknown'} "
        f"torch={_package_version('torch')} transformers={_package_version('transformers')}"
    )


def transcribe(audio_path: str | Path) -> str:
    """Transcribe an audio file to text. WAV is decoded in-process; other
    formats fall back to Whisper's own loader (needs ffmpeg)."""
    audio_path = Path(audio_path)
    asr = load_asr_model()
    if audio_path.suffix.lower() == ".wav":
        audio = load_audio(audio_path)
        result = asr(audio, batch_size=8)
    else:
        # mp3/m4a/... -> let transformers load it (requires ffmpeg in PATH)
        try:
            result = asr(str(audio_path), batch_size=8)
        except Exception as exc:
            raise ValueError(
                f"Cannot decode '{audio_path.name}'. Non-WAV formats need ffmpeg "
                f"in PATH. Detail: {exc}"
            ) from exc
    return str(result["text"]).strip()


# =============================================================================
# Step 2 - Summarize with an LLM (provider-swappable seam)
# =============================================================================


def build_prompt(transcript: str, section: str | None = None) -> str:
    """Build the summarization prompt.

    With ``section`` (an instruction) it builds a focused single-section
    prompt - the reliable path for a small local model, which extracts one
    focused list far better than three sections at once. Without it, the
    full three-section prompt (used by more capable providers). Role + clear
    instruction + required output format.
    """
    if section is not None:
        return (
            f"{section}\n\nMeeting transcript:\n{transcript}\n\nAnswer with only the bulleted list:"
        )
    requirements = "\n".join(
        f"{i}. {title}: {instr}" for i, (title, instr) in enumerate(SECTIONS, 1)
    )
    return (
        "You are an assistant that summarizes business meetings. From the "
        "transcript below, extract:\n" + requirements + f"\n\nTranscript:\n{transcript}\n"
    )


def stub(prompt: str) -> str:
    """The ``stub`` provider: needs no model at all. Echoes the first 60
    characters of the transcript embedded in a section prompt (whitespace
    collapsed), so the summary shows the transcript really reached the
    summarizer; a prompt without a transcript is echoed from its start."""
    marker = "Meeting transcript:\n"
    if marker in prompt:
        transcript = prompt.split(marker, 1)[1].split("\n\nAnswer with only", 1)[0]
        return "- (stub) " + " ".join(transcript.split())[:60]
    return "- (stub) " + prompt[:60]


def _summarize_ollama(prompt: str) -> str:
    """Generate with a model served by Ollama over its HTTP API.

    Uses only the standard library (urllib + json) - no extra dependency and
    no model weights in this repo's Python cache. Model and URL are read from
    the environment at call time (``MEETING_OLLAMA_MODEL``,
    ``MEETING_OLLAMA_URL``), not cached, so callers (and tests) can point at a
    different endpoint without reloading the module.
    """
    import json  # lazy import, see module docstring
    import urllib.request  # lazy import, see module docstring

    model = os.environ.get("MEETING_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    url = os.environ.get("MEETING_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": LLM_MAX_NEW_TOKENS},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("response", "")).strip()


def summarize_with_llm(prompt: str) -> str:
    """THE provider seam. Dispatches on ``MEETING_LLM_PROVIDER``:

        ollama (default) - a local model served by Ollama (no key, no weights
                           in this repo's cache). If unreachable, falls back
                           to the stub with a visible ``[ollama unavailable
                           ...]`` marker instead of raising.
        openai           - OpenAI Chat Completions (reads OPENAI_API_KEY and
                           MEETING_OPENAI_MODEL)
        local             - offline transformers causal LM fallback (weights
                           cached by Hugging Face, model from
                           MEETING_LLM_MODEL); use only if Ollama is
                           unavailable and no key is at hand
        stub             - echoes the start of the transcript (tests, no model)

    Swapping providers touches only this function's body (keys stay in the
    environment, never in code).
    """
    provider = os.environ.get("MEETING_LLM_PROVIDER", "ollama").lower()

    if provider == "stub":
        return stub(prompt)

    if provider == "openai":
        from openai import OpenAI  # lazy import, see module docstring

        client = OpenAI()  # reads OPENAI_API_KEY from the environment
        model = os.environ.get("MEETING_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "local":
        # Opt-in offline fallback: an instruction-tuned causal LM via
        # transformers, loaded once and run greedily. NOTE: this downloads
        # weights into the Hugging Face cache; prefer 'ollama' to keep them
        # off the system disk.
        global _LOCAL_LLM
        if _LOCAL_LLM is None:
            from transformers import (  # lazy heavy import, see module docstring
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            model_name = os.environ.get("MEETING_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL)
            revision = (
                DEFAULT_LOCAL_LLM_REVISION if model_name == DEFAULT_LOCAL_LLM_MODEL else "main"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            # Explicit `Any`: with transformers installed, `from_pretrained`'s
            # overloads would need a `# type: ignore` on some call sites;
            # without transformers installed (`ignore_missing_imports` makes
            # the import `Any` already), that ignore would be flagged as
            # unused. Widening the type here avoids the overload check in
            # either environment, so no ignore comment is needed.
            causal_model: Any = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
            _LOCAL_LLM = (tokenizer, causal_model)
        tokenizer, causal_model = _LOCAL_LLM
        messages = [
            {
                "role": "system",
                "content": "You are a concise meeting-notes assistant. Answer only with "
                "the requested bullet list, nothing else.",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        output = causal_model.generate(**inputs, max_new_tokens=LLM_MAX_NEW_TOKENS, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]  # strip prompt tokens
        return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()

    # default: Ollama, with a fallback to the stub if it cannot be reached.
    try:
        return _summarize_ollama(prompt)
    except OSError as exc:  # connection refused/timeout (urllib.error.URLError is an OSError)
        log.warning("ollama unavailable (%s); falling back to the stub", exc)
        return "[ollama unavailable - stub used] " + stub(prompt)


def summarize_structured(transcript: str) -> str:
    """Produce the three-section summary with one focused LLM call per section.

    Assembling under fixed ``## Header`` lines GUARANTEES the three sections
    exist regardless of model quality, and giving the small model one
    focused task at a time keeps each section reliably populated. Every call
    routes through the provider seam.
    """
    if not transcript.strip():
        transcript = "(empty transcript)"
    blocks = []
    for title, instr in SECTIONS:
        content = summarize_with_llm(build_prompt(transcript, section=instr))
        blocks.append(f"## {title}\n{content.strip() or '(unspecified)'}")
    return "\n\n".join(blocks)


# =============================================================================
# Chain + persistence
# =============================================================================


def _write_outputs(transcript: str, summary: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.txt").write_text(
        transcript.rstrip("\n") + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "summary.txt").write_text(
        summary.rstrip("\n") + "\n", encoding="utf-8", newline="\n"
    )
    log.info("wrote transcript.txt and summary.txt to %s", out_dir)


def process_meeting(audio_path: str | Path, out_dir: Path | None = None) -> dict[str, str]:
    """Full chain: audio -> transcript -> structured summary; saves both.

    Saves to ``out_dir``, by default ``output/runs/`` (gitignored): a real meeting's
    transcript must never overwrite the committed proofs, which only ``demo()`` writes.
    """
    transcript = transcribe(audio_path)
    summary = summarize_structured(transcript)
    _write_outputs(transcript, summary, OUT_DIR / "runs" if out_dir is None else out_dir)
    return {"transcript": transcript, "summary": summary}


# =============================================================================
# UI (Gradio), demo, CLI
# =============================================================================


def _ui_handler(audio_path: str | None) -> tuple[str, str]:
    """Gradio callback: returns (transcript, summary) or a friendly error."""
    if not audio_path:
        return "", "Upload an audio file first."
    try:
        result = process_meeting(audio_path)
        return result["transcript"], result["summary"]
    except Exception as exc:  # the details stay in the server log, not in the browser
        log.exception("could not process %s", Path(audio_path).name)
        return "", f"Could not process the file ({type(exc).__name__}); see the server log."


def build_ui() -> Any:
    """Construct the Gradio interface (no import side effects)."""
    import gradio as gr  # lazy heavy import, see module docstring

    return gr.Interface(
        fn=_ui_handler,
        inputs=gr.Audio(sources=["upload"], type="filepath", label="Meeting recording (.wav)"),
        outputs=[gr.Textbox(label="Transcript"), gr.Textbox(label="Structured summary")],
        title="Meeting Assistant (Whisper + LLM)",
        description="Whisper transcribes audio, then an LLM extracts topics, decisions "
        "and action items. Runs locally on CPU.",
        flagging_mode="never",
        analytics_enabled=False,
    )


def demo() -> DemoResult:
    """Transcribe a synthetic standup recording with real Whisper weights and
    summarize it with the ``stub`` provider.

    Tier ``models``: downloads/loads the real Whisper model and never runs in
    CI. The LLM provider is force-pinned to ``stub`` here (not "whichever
    provider is configured") - the default is ``ollama``, most machines have
    none running, and a machine that does would produce a different summary;
    a committed artefact must not depend on which services happen to be
    running. See the README for running the chain against a real provider.

    Generates (or reuses) ``data/standup.wav``, transcribes it, summarizes
    it, and writes a deterministic ``output/transcript.txt``,
    ``output/summary.txt`` and ``output/metrics.txt``. Only the console
    summary and the returned ``DemoResult`` carry timing; the three committed
    files never vary between runs.

    On a machine with no text-to-speech engine (e.g. a bare Linux runner
    without ``espeak-ng``), audio synthesis raises
    ``synthetic_audio.TTSEngineUnavailableError`` before any Whisper weights are
    touched; this is caught here and turned into a ``skipped`` result
    carrying an actionable note, never a raised exception or ``failed``
    status - a missing optional system package should not fail the gate.

    Synthesis succeeding and ``load_audio``/Whisper raising nothing is not
    proof the audio was any good. If the transcript's word count falls below
    ``MIN_TRANSCRIPT_WORD_RATIO`` of the synthesized script's own word count,
    this returns ``failed`` (never ``skipped`` - an implausibly short
    transcript from audio that parsed fine is treated as this project's own
    bug until proven otherwise), with a note carrying both word counts, the
    audio facts logged during synthesis, the ASR runtime facts
    (device/dtype/versions) and a truncated transcript excerpt. The three
    committed output files are written only after that check passes, so a
    failed run never overwrites them.
    """
    start = time.perf_counter()

    from projects.p10_meeting_assistant import synthetic_audio

    audio_facts: dict[str, synthetic_audio.AudioFacts] = {}
    try:
        # Only standup.wav: the demo never uses budget.wav, and not requesting it
        # sidesteps a real pyttsx3/eSpeak bug entirely - see generate()'s docstring.
        audio_paths = synthetic_audio.generate(
            DATA_DIR, only={"standup.wav"}, facts_out=audio_facts
        )
    except synthetic_audio.TTSEngineUnavailableError as exc:
        seconds = time.perf_counter() - start
        log.warning("demo: %s", exc)
        return DemoResult(
            "p10-meeting-assistant", "skipped", seconds=round(seconds, 2), note=str(exc)
        )
    audio_path = audio_paths["standup.wav"]

    previous_provider = os.environ.get("MEETING_LLM_PROVIDER")
    os.environ["MEETING_LLM_PROVIDER"] = "stub"
    try:
        transcript = transcribe(audio_path)
        summary = summarize_structured(transcript)
    finally:
        if previous_provider is None:
            os.environ.pop("MEETING_LLM_PROVIDER", None)
        else:
            os.environ["MEETING_LLM_PROVIDER"] = previous_provider

    words = len(transcript.split())
    script_words = len(synthetic_audio.SCRIPTS["standup.wav"].split())
    if words < MIN_TRANSCRIPT_WORD_RATIO * script_words:
        seconds = time.perf_counter() - start
        facts = audio_facts.get("standup.wav")
        facts_line = (
            facts.format("standup.wav")
            if facts is not None
            else "standup.wav: (no audio facts captured - reused from a previous run)"
        )
        asr_facts_line = _ASR_FACTS or "whisper: (no ASR runtime facts captured)"
        excerpt = transcript[:200].replace("\n", " ")
        note = (
            f"transcript has {words} word(s) but the synthesized script has {script_words}; "
            f"ratio {words / script_words:.2f} is below MIN_TRANSCRIPT_WORD_RATIO="
            f"{MIN_TRANSCRIPT_WORD_RATIO} - treating this as a broken/truncated audio file, "
            f"not a genuine transcription. {facts_line} {asr_facts_line} "
            f"transcript={excerpt!r}"
        )
        log.warning("demo: %s", note)
        return DemoResult("p10-meeting-assistant", "failed", seconds=round(seconds, 2), note=note)

    _write_outputs(transcript, summary, OUT_DIR)
    sections = summary.count("## ")
    metrics_lines = ["provider=stub", f"words={words}", f"sections={sections}"]
    (OUT_DIR / "metrics.txt").write_text(
        "\n".join(metrics_lines) + "\n", encoding="utf-8", newline="\n"
    )

    seconds = time.perf_counter() - start
    figures = {"provider": "stub", "words": str(words), "sections": str(sections)}
    log.info("demo: transcribed %d words into %d sections", words, sections)
    return DemoResult("p10-meeting-assistant", "ok", figures, seconds=round(seconds, 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("audio", nargs="?", type=Path, help="Path to an audio file (CLI mode).")
    p.add_argument("--demo", action="store_true", help="Run the demo (needs the models group).")
    p.add_argument("--port", type=int, default=7860, help="Port for the Gradio UI.")
    p.add_argument(
        "--host", default="127.0.0.1", help="Bind address for the Gradio UI (default: 127.0.0.1)."
    )
    p.add_argument("--verbose", action="store_true", help="Log at INFO level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: transcribe+summarize an audio file, run the demo, or launch the Gradio UI."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p10-meeting-assistant: {result.status}")
        if result.status in ("skipped", "failed"):
            if result.note:
                print(f"  note: {result.note}")
            return 0 if result.status == "skipped" else 1
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.audio:
        if not args.audio.is_file():
            print(f"File '{args.audio}' does not exist.")
            return 1
        chain_result = process_meeting(args.audio)
        words = len(chain_result["transcript"].split())
        sections = chain_result["summary"].count("## ")
        print(f"p10-meeting-assistant: {args.audio.name}")
        print(f"  words: {words}")
        print(f"  sections: {sections}")
        print(f"  wrote: {OUT_DIR / 'runs'}")
        return 0

    build_ui().launch(server_name=args.host, server_port=args.port, max_file_size="200mb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
