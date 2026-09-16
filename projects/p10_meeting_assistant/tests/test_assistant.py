"""Tests for assistant.py (Project P10).

The heavy models (Whisper, transformers LLMs) are never loaded here: the
transcription step and the LLM seam are monkeypatched, or the ``stub``
provider is forced via ``MEETING_LLM_PROVIDER``. Only the audio decoder
(scipy, no ffmpeg) runs for real, on a tiny WAV written in-test.

Run:
    uv run pytest projects/p10_meeting_assistant -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from projects.p10_meeting_assistant import assistant, synthetic_audio

# No blanket module-level ``pytestmark`` in this file: every test is marked
# individually (``core``) so a future real-Whisper test added here stays
# opt-in under ``models`` instead of being swept up by a blanket marker —
# matching projects/p08_image_captioning and projects/p09_chatbot. The
# sibling test_synthetic_audio.py does use a module-level ``core`` marker:
# nothing in it can ever need model weights.


def _write_wav(path: Path, sr: int, seconds: float = 1, stereo: bool = False) -> None:
    from scipy.io import wavfile

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    if stereo:
        tone = np.stack([tone, tone], axis=1)
    wavfile.write(path, sr, tone)


# =============================================================================
# load_audio
# =============================================================================


@pytest.mark.core
def test_resamples_to_16k_mono_float(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    _write_wav(p, sr=8000, seconds=1)  # 8 kHz -> expect ~16 kHz
    audio = assistant.load_audio(p)
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # mono
    assert abs(len(audio) - 16000) <= 64
    assert float(np.max(np.abs(audio))) <= 1.0 + 1e-6


@pytest.mark.core
def test_stereo_is_downmixed(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    _write_wav(p, sr=16000, seconds=1, stereo=True)
    assert assistant.load_audio(p).ndim == 1


@pytest.mark.core
def test_8_bit_unsigned_wav_is_centred_and_scaled(tmp_path: Path) -> None:
    """8-bit WAV is unsigned: 128 is silence, 0 and 255 are the extremes.
    Dividing by 255 without centring would map silence to about +0.5."""
    from scipy.io import wavfile

    p = tmp_path / "u8.wav"
    wavfile.write(p, 16000, np.array([128, 0, 255, 128], dtype=np.uint8))
    audio = assistant.load_audio(p)
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, -1.0, 127 / 128, 0.0])


@pytest.mark.core
def test_non_wav_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError):
        assistant.load_audio(p)


# =============================================================================
# build_prompt
# =============================================================================


@pytest.mark.core
def test_combined_prompt_has_transcript_and_all_sections() -> None:
    prompt = assistant.build_prompt("ACME transcript body")
    assert "ACME transcript body" in prompt
    for title, _ in assistant.SECTIONS:
        assert title in prompt


@pytest.mark.core
def test_focused_prompt_includes_instruction_and_transcript() -> None:
    prompt = assistant.build_prompt("body text", section="List the decisions.")
    assert "List the decisions." in prompt
    assert "body text" in prompt


# =============================================================================
# summarize_structured — the provider seam is monkeypatched
# =============================================================================


@pytest.mark.core
def test_structured_summary_has_three_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant, "summarize_with_llm", lambda prompt: "- example content")
    summary = assistant.summarize_structured("some transcript")
    for title, _ in assistant.SECTIONS:
        assert f"## {title}" in summary


@pytest.mark.core
def test_calls_llm_once_per_section(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return "- x"

    monkeypatch.setattr(assistant, "summarize_with_llm", fake)
    assistant.summarize_structured("t")
    assert len(calls) == len(assistant.SECTIONS)


@pytest.mark.core
def test_empty_content_falls_back_to_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant, "summarize_with_llm", lambda prompt: "   ")
    summary = assistant.summarize_structured("t")
    assert "(unspecified)" in summary


# =============================================================================
# summarize_with_llm — provider dispatch
# =============================================================================


@pytest.mark.core
def test_stub_provider_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "stub")
    out = assistant.summarize_with_llm("any prompt")
    assert "stub" in out.lower()


@pytest.mark.core
def test_missing_ollama_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MEETING_OLLAMA_URL", "http://127.0.0.1:9")  # closed port
    text = assistant.summarize_with_llm("prompt")
    assert text.startswith("[ollama unavailable")  # falls back to the stub with a visible note


# =============================================================================
# process_meeting — chain + persistence
# =============================================================================


@pytest.mark.core
def test_process_meeting_writes_both_files(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assistant, "transcribe", lambda audio_path: "fake meeting transcript")
    monkeypatch.setattr(assistant, "summarize_with_llm", lambda prompt: "- point")
    monkeypatch.setattr(assistant, "OUT_DIR", tmp_output)

    result = assistant.process_meeting("whichever.wav")

    assert result["transcript"] == "fake meeting transcript"
    assert (tmp_output / "transcript.txt").exists()
    assert (tmp_output / "summary.txt").exists()
    summary_text = (tmp_output / "summary.txt").read_text(encoding="utf-8")
    for title, _ in assistant.SECTIONS:
        assert f"## {title}" in summary_text


@pytest.mark.core
def test_stub_pipeline_produces_all_three_sections(
    tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "stub")
    monkeypatch.setattr(assistant, "OUT_DIR", tmp_output)
    monkeypatch.setattr(
        assistant, "transcribe", lambda path: "We agreed to ship on Friday. Ana owns the release."
    )
    result = assistant.process_meeting("ignored.wav")
    summary = (tmp_output / "summary.txt").read_text(encoding="utf-8")
    for header in ("Topics discussed", "Decisions", "Action items"):
        assert header in summary and header in result["summary"]


@pytest.mark.core
def test_stub_summary_is_derived_from_the_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every section prompt starts with a long instruction, so a stub that
    echoed only the start of the prompt would give the same summary for any
    transcript. Two different transcripts must give two different summaries,
    each echoing its own transcript."""
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "stub")
    first = assistant.summarize_structured("We agreed to ship on Friday.")
    second = assistant.summarize_structured("The budget review moved to March.")
    assert first != second
    assert "We agreed to ship on Friday." in first
    assert "The budget review moved to March." in second


# =============================================================================
# demo() — must degrade to "skipped", never "failed", without a speech engine
# =============================================================================


@pytest.mark.core
def test_demo_skips_cleanly_when_tts_engine_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a bare CI runner (no espeak-ng/SAPI5), ``synthetic_audio.generate``
    raises ``TTSEngineUnavailableError`` before any Whisper weights are touched.
    ``demo()`` must turn that into a ``skipped`` DemoResult carrying an
    actionable note (not a raised exception, and not ``failed`` — which
    would make ``shared.demo.main()`` exit 1 on a machine that is merely
    missing an optional system package).
    """

    def fake_generate(
        data_dir: Path,
        force: bool = False,
        only: object = None,
        facts_out: dict[str, synthetic_audio.AudioFacts] | None = None,
    ) -> dict[str, Path]:
        raise synthetic_audio.TTSEngineUnavailableError(
            "No text-to-speech engine is available (pyttsx3.init() failed: boom); "
            "install it with `sudo apt-get install espeak-ng` (or your distro's equivalent)."
        )

    monkeypatch.setattr(synthetic_audio, "generate", fake_generate)
    result = assistant.demo()
    assert result.status == "skipped"
    assert "espeak-ng" in result.note


# =============================================================================
# demo() — an implausibly short transcript must fail, not "ok"
# =============================================================================


def _fake_generate_returning(wav_path: Path, facts: synthetic_audio.AudioFacts) -> object:
    """Builds a ``synthetic_audio.generate``-compatible stand-in that skips
    pyttsx3/AIFF conversion entirely and hands back a fixed path plus facts,
    matching ``generate()``'s real ``facts_out`` output-parameter contract."""

    def fake_generate(
        data_dir: Path,
        force: bool = False,
        only: object = None,
        facts_out: dict[str, synthetic_audio.AudioFacts] | None = None,
    ) -> dict[str, Path]:
        if facts_out is not None:
            facts_out["standup.wav"] = facts
        return {"standup.wav": wav_path}

    return fake_generate


@pytest.mark.core
def test_demo_fails_when_transcript_is_implausibly_short(
    tmp_path: Path, tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audio that parses and transcribes without raising, but yields only 4
    of the synthesized script's 73 words, must not report "ok". A wrong
    threshold, or a guard that never fires, would let this test pass with
    "ok" instead."""
    wav_path = tmp_path / "standup.wav"
    wav_path.write_bytes(b"not real audio - transcribe() is stubbed below")
    facts = synthetic_audio.AudioFacts(
        container="AIFF",
        compression="NONE",
        channels=1,
        sample_width=2,
        sample_rate=22050,
        declared_frames=500,
        actual_frames=20,
        duration_seconds=20 / 22050,
        file_size_bytes=4000,
    )
    monkeypatch.setattr(synthetic_audio, "generate", _fake_generate_returning(wav_path, facts))
    monkeypatch.setattr(assistant, "OUT_DIR", tmp_output)
    monkeypatch.setattr(assistant, "transcribe", lambda path: "only four words here")
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "stub")

    result = assistant.demo()

    assert result.status == "failed"
    assert "4" in result.note  # the transcript's own word count
    assert "73" in result.note  # SCRIPTS["standup.wav"]'s word count
    assert "AIFF" in result.note  # the audio facts made it into the note
    assert "only four words here" in result.note  # transcript excerpt
    # A failed demo must not overwrite any committed, byte-identical-across-runs
    # proof with output from broken audio.
    assert not (tmp_output / "metrics.txt").exists()
    assert not (tmp_output / "transcript.txt").exists()
    assert not (tmp_output / "summary.txt").exists()


@pytest.mark.core
def test_demo_still_ok_when_transcript_word_count_is_plausible(
    tmp_path: Path, tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The companion to the test above: a transcript at/above
    ``MIN_TRANSCRIPT_WORD_RATIO`` of the script's word count (here, a stand-in
    transcript with plenty of words) must still report "ok" and write
    metrics.txt — this is the Windows/Linux baseline the guard must not
    break. A guard that always fires (or a threshold set too high) would
    turn this into "failed" instead."""
    wav_path = tmp_path / "standup.wav"
    wav_path.write_bytes(b"not real audio - transcribe() is stubbed below")
    facts = synthetic_audio.AudioFacts(
        container="RIFF",
        compression="PCM",
        channels=1,
        sample_width=2,
        sample_rate=22050,
        declared_frames=200000,
        actual_frames=200000,
        duration_seconds=200000 / 22050,
        file_size_bytes=400044,
    )
    monkeypatch.setattr(synthetic_audio, "generate", _fake_generate_returning(wav_path, facts))
    monkeypatch.setattr(assistant, "OUT_DIR", tmp_output)
    plausible_transcript = " ".join(["word"] * 80)  # >= 73 * 0.5
    monkeypatch.setattr(assistant, "transcribe", lambda path: plausible_transcript)
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "stub")

    result = assistant.demo()

    assert result.status == "ok"
    assert result.figures["words"] == "80"
    metrics = (tmp_output / "metrics.txt").read_text(encoding="utf-8")
    assert "words=80" in metrics
    assert (tmp_output / "transcript.txt").read_text(
        encoding="utf-8"
    ) == plausible_transcript + "\n"
    assert "## Action items" in (tmp_output / "summary.txt").read_text(encoding="utf-8")


# =============================================================================
# load_asr_model — Whisper pinned to CPU/float32, and the device it actually
# landed on is logged and carried into the failure note
# =============================================================================


class _FakeModel:
    """Stands in for a loaded Whisper model. Its device/dtype deliberately
    differ from the arguments ``load_asr_model`` passes, so a facts line
    that merely echoed those arguments (instead of reading the model back)
    would fail the assertions below."""

    device = "fake-readback-device:7"
    dtype = "fake.float32-readback"


class _FakePipeline:
    model = _FakeModel()


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[tuple[object, ...], dict[str, object]]]
) -> None:
    """Replace ``transformers`` (installed or not — the core env has no
    transformers) with a module whose ``pipeline`` records its arguments and
    returns ``_FakePipeline`` — no model download, no torch. Also resets the
    cached pipeline/facts and swaps in a throwaway ASR-facts logger so the
    handler ``load_asr_model`` attaches never leaks into other tests."""
    import logging
    import sys
    import types

    def fake_pipeline(*args: object, **kwargs: object) -> _FakePipeline:
        calls.append((args, kwargs))
        return _FakePipeline()

    fake_module = types.ModuleType("transformers")
    fake_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    monkeypatch.setattr(assistant, "_ASR", None)
    monkeypatch.setattr(assistant, "_ASR_FACTS", None)
    monkeypatch.setattr(assistant, "asr_log", logging.Logger("test-p10-asr-facts"))


@pytest.mark.core
def test_load_asr_model_pins_cpu_and_float32(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the explicit pin, transformers may auto-place Whisper on an
    accelerator (Apple's MPS on Apple Silicon, where it transcribed healthy
    audio as noise). Removing either keyword, or changing its value, fails
    this test."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    _install_fake_transformers(monkeypatch, calls)

    asr = assistant.load_asr_model()

    assert isinstance(asr, _FakePipeline)
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["device"] == "cpu"
    assert kwargs["dtype"] == "float32"
    assert "torch_dtype" not in kwargs  # deprecated in transformers 5
    # The facts line is printed (visible without any logging configuration,
    # as under `uv run demo --all`) and reads the model back.
    out = capsys.readouterr().out
    assert "device=fake-readback-device:7" in out
    assert "dtype=fake.float32-readback" in out
    assert "torch=" in out and "transformers=" in out


@pytest.mark.core
def test_demo_failure_note_includes_asr_device_facts(
    tmp_path: Path, tmp_output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transcript guard's failure note must carry the device/dtype
    Whisper actually ran on (read back from the loaded model) next to the
    audio facts, so the next macOS failure says where transcription ran."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    _install_fake_transformers(monkeypatch, calls)
    wav_path = tmp_path / "standup.wav"
    wav_path.write_bytes(b"not real audio - transcribe() is stubbed below")
    facts = synthetic_audio.AudioFacts(
        container="AIFC",
        compression="twos",
        channels=1,
        sample_width=2,
        sample_rate=22050,
        declared_frames=608937,
        actual_frames=608937,
        duration_seconds=608937 / 22050,
        file_size_bytes=1222000,
    )
    monkeypatch.setattr(synthetic_audio, "generate", _fake_generate_returning(wav_path, facts))
    monkeypatch.setattr(assistant, "OUT_DIR", tmp_output)

    def fake_transcribe(path: object) -> str:
        assistant.load_asr_model()  # the real loader, against the fake transformers
        return "il Tottenham ca cagggg"

    monkeypatch.setattr(assistant, "transcribe", fake_transcribe)
    monkeypatch.setenv("MEETING_LLM_PROVIDER", "stub")

    result = assistant.demo()

    assert result.status == "failed"
    assert "AIFC" in result.note  # audio facts still present
    assert "device=fake-readback-device:7" in result.note
    assert "dtype=fake.float32-readback" in result.note
    assert "torch=" in result.note and "transformers=" in result.note
