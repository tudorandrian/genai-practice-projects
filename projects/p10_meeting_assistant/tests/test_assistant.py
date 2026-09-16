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

# No blanket module-level ``pytestmark`` here: every test is marked
# individually (``core``) so a future real-Whisper test added to this file
# stays opt-in under ``models`` instead of being swept up by a blanket
# marker — matching projects/p08_image_captioning and projects/p09_chatbot.


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

    def fake_generate(data_dir: Path, force: bool = False) -> dict[str, Path]:
        raise synthetic_audio.TTSEngineUnavailableError(
            "No text-to-speech engine is available (pyttsx3.init() failed: boom); "
            "install it with `sudo apt-get install espeak-ng` (or your distro's equivalent)."
        )

    monkeypatch.setattr(synthetic_audio, "generate", fake_generate)
    result = assistant.demo()
    assert result.status == "skipped"
    assert "espeak-ng" in result.note
