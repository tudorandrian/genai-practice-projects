"""Tests for synthetic_audio.py (Project P10).

pyttsx3 is never imported for real here: ``generate()`` imports it lazily
inside ``_init_engine()``, so stuffing a fake module into ``sys.modules``
before calling ``generate()`` is enough - these tests run offline, without
the ``models`` dependency group, like the rest of this project's ``core``
suite.

Run:
    uv run pytest projects/p10_meeting_assistant -q
"""

from __future__ import annotations

import logging
import math
import random
import struct
import sys
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import wavfile

from projects.p10_meeting_assistant import synthetic_audio

pytestmark = pytest.mark.core


class _FakeEngine:
    """Stands in for a pyttsx3 driver. Any file name in ``silent_names`` reproduces the
    real bug this module works around: ``save_to_file``/``runAndWait`` both report
    success but no bytes ever reach disk for that specific file (the eSpeak
    weak-reference callback failure) - decided by the path's own name, not by call
    order, since ``generate()`` iterates a ``set`` whose order is not guaranteed."""

    def __init__(self, silent_names: set[str], write_bytes: bytes | None = None) -> None:
        self._silent_names = silent_names
        self._write_bytes = write_bytes or b"RIFF....WAVEfmt fake audio bytes"
        self._pending: tuple[str, str] | None = None

    def setProperty(self, _name: str, _value: object) -> None:  # noqa: N802 - pyttsx3's real method name
        pass

    def save_to_file(self, text: str, path: str) -> None:
        self._pending = (text, path)

    def runAndWait(self) -> None:  # noqa: N802 - pyttsx3's real method name
        if self._pending is None:
            return
        _text, path = self._pending
        if Path(path).name not in self._silent_names:
            Path(path).write_bytes(self._write_bytes)


def _fake_pyttsx3(silent_names: set[str], write_bytes: bytes | None = None) -> Any:
    """A fake ``pyttsx3`` module: every engine it hands out silently drops any file
    whose name is in ``silent_names`` and writes ``write_bytes`` (default: fake WAV
    bytes) for every other file."""

    class _Module:
        @staticmethod
        def init() -> _FakeEngine:
            return _FakeEngine(silent_names, write_bytes)

    return _Module


# =============================================================================
# Hand-built AIFF/AIFF-C byte fixtures (no aifc module - removed in 3.13; no
# binary fixture files - built in code, per the mandatory constraints).
# =============================================================================


def _write_ieee_extended(value: float) -> bytes:
    """Encode a big-endian 80-bit IEEE-754 extended-precision float - the
    inverse of ``synthetic_audio._read_ieee_extended`` - so a COMM chunk's
    sample rate can be hand-built for these fixtures."""
    sign_bit = 0x8000 if value < 0 else 0
    value = abs(value)
    if value == 0:
        expon, himant, lomant = 0, 0, 0
    else:
        fmant, expon = math.frexp(value)
        expon += 16382
        if expon < 0:
            fmant = math.ldexp(fmant, expon)
            expon = 0
        expon |= sign_bit
        fmant = math.ldexp(fmant, 32)
        fsmant = math.floor(fmant)
        himant = int(fsmant)
        fmant = math.ldexp(fmant - fsmant, 32)
        lomant = int(math.floor(fmant))
    return struct.pack(">HII", expon, himant, lomant)


def _chunk(chunk_id: bytes, body: bytes) -> bytes:
    padded = body + (b"\x00" if len(body) % 2 else b"")
    return chunk_id + struct.pack(">I", len(body)) + padded


def _build_aiff_or_aifc(
    samples: list[int],
    *,
    channels: int = 1,
    sample_width: int = 2,
    frame_rate: int = 22050,
    aifc_compression: bytes | None = None,
    actual_byte_order: str | None = None,
) -> bytes:
    """Hand-build a minimal AIFF (``aifc_compression=None``) or AIFF-C
    (``aifc_compression=b"NONE"``/``b"sowt"``/anything else) file with one
    COMM chunk and one SSND chunk wrapping ``samples`` (signed integers, one
    per sample_width-sized slot; widths 1, 2, 3 and 4). For AIFF-C 'sowt',
    samples are written little-endian (as a real 'sowt' file is); every other
    case is written big-endian, matching real AIFF/AIFF-C bytes on disk.

    ``actual_byte_order`` (``"<"`` or ``">"``) overrides the byte order the
    *data* is written in, independent of ``aifc_compression``'s *label*, to
    build a file whose header and samples disagree.
    """
    form_type = b"AIFF" if aifc_compression is None else b"AIFC"
    frame_count = len(samples) // channels

    comm_body = struct.pack(">hlh", channels, frame_count, sample_width * 8) + _write_ieee_extended(
        float(frame_rate)
    )
    if aifc_compression is not None:
        comm_body += aifc_compression + b"\x00"  # + zero-length Pascal compression name

    if actual_byte_order is not None:
        byte_order = actual_byte_order
    else:
        byte_order = "<" if aifc_compression in (b"sowt", b"SOWT") else ">"
    if sample_width == 3:  # struct has no 24-bit code
        little = byte_order == "<"
        pcm = b"".join((s & 0xFFFFFF).to_bytes(3, "little" if little else "big") for s in samples)
    else:
        width_code = {1: "b", 2: "h", 4: "l"}[sample_width]
        pcm = struct.pack(f"{byte_order}{len(samples)}{width_code}", *samples)

    ssnd_body = struct.pack(">II", 0, 0) + pcm  # offset=0, blockSize=0

    body = form_type + _chunk(b"COMM", comm_body) + _chunk(b"SSND", ssnd_body)
    return b"FORM" + struct.pack(">I", len(body)) + body


def _disable_stabilize_and_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """For tests exercising generate()'s plumbing (existence checks, caching,
    conversion correctness) rather than its stabilization wait or
    plausibility floor: the fake engines here write a handful of bytes/
    samples instantly and synchronously, so there is nothing to actually
    wait for, and the fixtures are far too short to clear
    MIN_SECONDS_PER_WORD's real-world floor. Disabling both keeps those
    tests fast (no real sleep) and focused on what they mean to test -
    stabilization and the floor get their own dedicated tests below."""
    monkeypatch.setattr(synthetic_audio, "STABILIZE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(synthetic_audio, "MIN_SECONDS_PER_WORD", 0.0)


def test_generate_writes_only_the_requested_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_stabilize_and_floor(monkeypatch)
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_pyttsx3(silent_names=set()))
    paths = synthetic_audio.generate(tmp_path, only={"standup.wav"})
    assert set(paths) == {"standup.wav"}
    assert paths["standup.wav"].exists()
    assert not (tmp_path / "budget.wav").exists()


def test_generate_raises_when_pyttsx3_silently_drops_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pyttsx3 can report success from
    ``save_to_file``/``runAndWait`` while never writing bytes to the requested path
    (the real eSpeak weak-reference bug this module works around, reproduced here
    without a real TTS engine). ``generate()`` must not return that missing path as
    if it had succeeded - it must raise the same ``TTSEngineUnavailableError`` a
    genuinely absent engine raises, so callers already handling "no engine at all"
    (``demo()``) also handle "engine present but silently incomplete" the same way:
    a clean ``skipped`` result, never a ``failed`` one from a confusing
    two-steps-removed ``ValueError`` in Whisper.
    """
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_pyttsx3(silent_names={"standup.wav"}))
    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="standup.wav"):
        synthetic_audio.generate(tmp_path, only={"standup.wav", "budget.wav"})


def test_generate_succeeds_when_every_requested_file_is_actually_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_stabilize_and_floor(monkeypatch)
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_pyttsx3(silent_names=set()))
    paths = synthetic_audio.generate(tmp_path, only={"standup.wav", "budget.wav"})
    assert set(paths) == {"standup.wav", "budget.wav"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())


def test_generate_reuses_existing_files_without_calling_pyttsx3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "standup.wav").write_bytes(b"already here")

    # Any "import pyttsx3" now raises ImportError (sys.modules["pyttsx3"] = None is
    # Python's own way to force that) - proving generate() never reaches the import
    # at all when there is nothing pending to synthesize.
    monkeypatch.setitem(sys.modules, "pyttsx3", None)
    paths = synthetic_audio.generate(tmp_path, only={"standup.wav"})
    assert paths["standup.wav"].read_bytes() == b"already here"


def test_generate_converts_a_macos_aiff_file_written_by_a_fake_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the real bug end to end without a real TTS engine: a fake
    engine standing in for macOS's NSSpeechSynthesizer driver writes AIFF
    bytes to the ``.wav`` path; ``generate()`` must hand back a file that is
    actually readable as a WAV, not the mislabelled AIFF bytes verbatim."""
    _disable_stabilize_and_floor(monkeypatch)
    aiff_bytes = _build_aiff_or_aifc([100, -100, 200, -200], frame_rate=22050)
    monkeypatch.setitem(
        sys.modules, "pyttsx3", _fake_pyttsx3(silent_names=set(), write_bytes=aiff_bytes)
    )
    paths = synthetic_audio.generate(tmp_path, only={"standup.wav"})
    rate, data = wavfile.read(paths["standup.wav"])
    assert rate == 22050
    assert data.tolist() == [100, -100, 200, -200]


# =============================================================================
# generate() - wait for asynchronous writes, plausibility floor
# =============================================================================
#
# macOS's NSSpeechSynthesizer writes asynchronously and runAndWait() returns
# before the write finishes; read too early, the file is a header plus a few
# milliseconds of audio. These tests drive generate()'s stabilization wait and
# plausibility floor without any real TTS engine or real sleeping
# (sleep/monotonic are injected).


class _AsyncWriteEngine:
    """Stands in for NSSpeechSynthesizer: ``runAndWait()`` writes
    ``initial_bytes`` synchronously (so the exists/non-empty check right
    after it still passes) - the rest of a real asynchronous write arrives
    later, driven here by whatever the test wires up to run on each
    simulated "tick" (the injected ``sleep``), modeling the engine finishing
    the write only after ``runAndWait()`` has already returned."""

    def __init__(self, path_holder: dict[str, str], initial_bytes: bytes) -> None:
        self._path_holder = path_holder
        self._initial_bytes = initial_bytes

    def setProperty(self, _name: str, _value: object) -> None:  # noqa: N802 - pyttsx3's real method name
        pass

    def save_to_file(self, _text: str, path: str) -> None:
        self._path_holder["path"] = path

    def runAndWait(self) -> None:  # noqa: N802 - pyttsx3's real method name
        Path(self._path_holder["path"]).write_bytes(self._initial_bytes)


def _fake_async_pyttsx3(initial_bytes: bytes) -> Any:
    path_holder: dict[str, str] = {}

    class _Module:
        @staticmethod
        def init() -> _AsyncWriteEngine:
            return _AsyncWriteEngine(path_holder, initial_bytes)

    return _Module


def test_generate_waits_for_an_asynchronously_written_file_before_converting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file grows across several polls (delivered by the injected
    ``sleep`` callback, standing in for the passage of real time) before
    settling on the complete AIFF bytes. generate() must not read/convert
    the file before its size stops changing - converting after the first
    poll would only see a fraction of the frames, which is exactly what the
    final assertion below would catch."""
    monkeypatch.setattr(synthetic_audio, "MIN_SECONDS_PER_WORD", 0.0)
    samples = list(range(30))
    full_aiff = _build_aiff_or_aifc(samples, channels=1, sample_width=2, frame_rate=22050)
    chunk_sizes = [len(full_aiff) // 3, (len(full_aiff) * 2) // 3, len(full_aiff)]

    # runAndWait() writes the first chunk synchronously; the rest grows in
    # across the next two simulated polls, then stops.
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_async_pyttsx3(full_aiff[: chunk_sizes[0]]))

    target = tmp_path / "standup.wav"
    progress = {"n": 0}

    def fake_sleep(_seconds: float) -> None:
        progress["n"] += 1
        idx = min(progress["n"], len(chunk_sizes) - 1)
        target.write_bytes(full_aiff[: chunk_sizes[idx]])

    tick = {"t": 0.0}

    def fake_monotonic() -> float:
        tick["t"] += synthetic_audio.STABILIZE_POLL_SECONDS
        return tick["t"]

    paths = synthetic_audio.generate(
        tmp_path, only={"standup.wav"}, sleep=fake_sleep, monotonic=fake_monotonic
    )

    rate, data = wavfile.read(paths["standup.wav"])
    assert rate == 22050
    assert data.tolist() == samples  # every frame present -- not converted mid-write


def test_generate_raises_if_the_file_never_stops_growing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathological case: the file keeps changing size past the timeout
    and never settles. generate() must give up and raise
    TTSEngineUnavailableError (so demo() can degrade to skipped) rather than
    hang or silently accept a moving target."""
    monkeypatch.setattr(synthetic_audio, "MIN_SECONDS_PER_WORD", 0.0)
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_async_pyttsx3(b"x" * 16))

    target = tmp_path / "standup.wav"
    counter = {"n": 0}

    def fake_sleep(_seconds: float) -> None:
        counter["n"] += 1
        target.write_bytes(b"x" * (16 + counter["n"]))  # a different size every call

    tick = {"t": 0.0}

    def fake_monotonic() -> float:
        tick["t"] += synthetic_audio.STABILIZE_POLL_SECONDS
        return tick["t"]

    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="never finished writing"):
        synthetic_audio.generate(
            tmp_path, only={"standup.wav"}, sleep=fake_sleep, monotonic=fake_monotonic
        )


class _FakeClock:
    """An injectable ``sleep``/``monotonic`` pair: ``sleep`` advances time by
    exactly the requested seconds and runs ``on_sleep`` (the simulated
    engine's progress); ``monotonic`` just reads the time. No real sleeping."""

    def __init__(self, on_sleep: Any = None) -> None:
        self.now = 1000.0
        self.sleeps = 0
        self._on_sleep = on_sleep

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds
        if self._on_sleep is not None:
            self._on_sleep(self.sleeps)

    def monotonic(self) -> float:
        return self.now


def _speech_like_samples(seconds: float, sample_rate: int = 22050) -> list[int]:
    """A smooth, speech-amplitude signal (so the byte-order check is
    conclusive and the WAV round-trips exactly), ``seconds`` long."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    wave_ = 8000 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return [int(v) for v in wave_.astype(np.int16)]


# The engine can leave the file at its header-only size (118 frames, ~5 ms)
# for longer than the STABILIZE_REQUIRED_STABLE_READS x STABILIZE_POLL_SECONDS
# window before any audio arrives. These tests use the real
# MIN_SECONDS_PER_WORD floor (73 words -> 7.3s for standup.wav).
_HEADER_ONLY_FRAMES = 118


def test_generate_keeps_waiting_while_a_stable_file_is_below_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is header-only and size-stable for several polls *longer*
    than the stability window, then grows to the full recording and stops.
    It must be accepted with every frame. A wait that declared the
    header-only file final after STABILIZE_REQUIRED_STABLE_READS reads would
    then reject it as implausibly short, and this test would raise."""
    full_samples = _speech_like_samples(8.0)  # above the 7.3s floor
    full = _build_aiff_or_aifc(full_samples, aifc_compression=b"twos")
    header_only = _build_aiff_or_aifc(full_samples[:_HEADER_ONLY_FRAMES], aifc_compression=b"twos")
    target = tmp_path / "standup.wav"
    quiet_polls = synthetic_audio.STABILIZE_REQUIRED_STABLE_READS + 5

    def engine_progress(sleeps: int) -> None:
        if sleeps == quiet_polls:
            target.write_bytes(full)  # the first (and only) audio write arrives late

    clock = _FakeClock(engine_progress)
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_async_pyttsx3(header_only))
    facts_out: dict[str, synthetic_audio.AudioFacts] = {}

    paths = synthetic_audio.generate(
        tmp_path,
        only={"standup.wav"},
        facts_out=facts_out,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    rate, data = wavfile.read(paths["standup.wav"])
    assert rate == 22050
    assert data.tolist() == full_samples  # every frame, not the header-only file
    assert facts_out["standup.wav"].actual_frames == len(full_samples)
    # accepted once the full file had been stable for the required reads
    assert clock.sleeps == quiet_polls + synthetic_audio.STABILIZE_REQUIRED_STABLE_READS - 1


def test_generate_times_out_waiting_for_audio_on_a_header_only_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A file that stays header-only for the whole STABILIZE_TIMEOUT_SECONDS
    must raise with "timed out" wording, the elapsed time and the audio
    facts, and only after the full timeout, not after the stability window."""
    header_only = _build_aiff_or_aifc(
        _speech_like_samples(8.0)[:_HEADER_ONLY_FRAMES], aifc_compression=b"twos"
    )
    clock = _FakeClock()
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_async_pyttsx3(header_only))

    with (
        caplog.at_level(logging.INFO, logger=synthetic_audio.log.name),
        pytest.raises(
            synthetic_audio.TTSEngineUnavailableError,
            match="timed out after 60.0s waiting for audio",
        ) as excinfo,
    ):
        synthetic_audio.generate(
            tmp_path, only={"standup.wav"}, sleep=clock.sleep, monotonic=clock.monotonic
        )

    message = str(excinfo.value)
    assert "container=AIFC" in message  # the audio facts made it into the message
    assert f"actual={_HEADER_ONLY_FRAMES}" in message
    assert "7.300s minimum" in message
    expected_polls = (
        int(synthetic_audio.STABILIZE_TIMEOUT_SECONDS / synthetic_audio.STABILIZE_POLL_SECONDS) + 1
    )
    assert f"polls={expected_polls}" in message
    # the facts log line carries the wait's elapsed time and poll count too
    facts_lines = [r.getMessage() for r in caplog.records if r.name == synthetic_audio.log.name]
    assert any("wait=60.00s" in line and f"polls={expected_polls}" in line for line in facts_lines)


def test_generate_deletes_the_file_when_waiting_for_audio_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate() skips any script whose file already exists, so a
    header-only file left behind by a timed-out synthesis would be reused by
    the next call and fail later in load_audio. It must be deleted before
    the error propagates, so the next call synthesizes again."""
    header_only = _build_aiff_or_aifc(
        _speech_like_samples(8.0)[:_HEADER_ONLY_FRAMES], aifc_compression=b"twos"
    )
    clock = _FakeClock()
    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_async_pyttsx3(header_only))

    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="timed out"):
        synthetic_audio.generate(
            tmp_path, only={"standup.wav"}, sleep=clock.sleep, monotonic=clock.monotonic
        )

    assert not (tmp_path / "standup.wav").exists()


def test_generate_spares_a_finished_sibling_when_the_other_script_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two scripts are requested in one generate() call. Whichever of the two
    ``_finish_synthesized_files`` processes first finishes cleanly and must
    survive; the other keeps growing until it times out. generate()'s
    cleanup on TTSEngineUnavailableError must delete only the failed file,
    not a sibling that had already finished and been added to ``finished``
    earlier in the same call."""
    monkeypatch.setattr(synthetic_audio, "MIN_SECONDS_PER_WORD", 0.0)
    names = {"standup.wav", "budget.wav"}
    # _finish_synthesized_files iterates `pending`, built from `set(only)` -
    # its order is hash-dependent, not insertion order (see _FakeEngine's
    # docstring above). Read the real order back so the "good" role always
    # lands on whichever name is actually processed first.
    good_name, bad_name = list(set(names))

    good_samples = [1, 2, 3, 4]
    good_bytes = _build_aiff_or_aifc(good_samples, channels=1, sample_width=2, frame_rate=22050)
    bad_growth = {"n": 0}

    class _TwoScriptEngine:
        def __init__(self) -> None:
            self._pending_path: str | None = None

        def setProperty(self, _name: str, _value: object) -> None:  # noqa: N802
            pass

        def save_to_file(self, _text: str, path: str) -> None:
            self._pending_path = path

        def runAndWait(self) -> None:  # noqa: N802
            assert self._pending_path is not None
            path = Path(self._pending_path)
            if path.name == good_name:
                path.write_bytes(good_bytes)
            else:
                path.write_bytes(b"x" * 16)

    class _Module:
        @staticmethod
        def init() -> _TwoScriptEngine:
            return _TwoScriptEngine()

    monkeypatch.setitem(sys.modules, "pyttsx3", _Module)

    def fake_sleep(_seconds: float) -> None:
        # Only the bad file's wait loop should ever observe a size change;
        # the good file's bytes, written once above, never change again.
        bad_growth["n"] += 1
        (tmp_path / bad_name).write_bytes(b"x" * (16 + bad_growth["n"]))

    tick = {"t": 0.0}

    def fake_monotonic() -> float:
        tick["t"] += synthetic_audio.STABILIZE_POLL_SECONDS
        return tick["t"]

    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="never finished writing"):
        synthetic_audio.generate(tmp_path, only=names, sleep=fake_sleep, monotonic=fake_monotonic)

    assert not (tmp_path / bad_name).exists()
    good_path = tmp_path / good_name
    assert good_path.exists()
    rate, data = wavfile.read(good_path)
    assert rate == 22050
    assert data.tolist() == good_samples  # sibling file is intact, not truncated or corrupted


def test_generate_accepts_an_already_final_file_after_minimum_stable_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case (Windows/Linux): the file is already complete by the
    time generate() looks at it. It
    must still be accepted quickly - via the minimum number of stable
    reads, not the full timeout - costing exactly
    STABILIZE_REQUIRED_STABLE_READS - 1 sleeps. A stabilization loop that
    accepts on the very first read (no confirmation at all) or that keeps
    polling past the point of confidence would make this assertion fail."""
    monkeypatch.setattr(synthetic_audio, "MIN_SECONDS_PER_WORD", 0.0)
    samples = [1, 2, 3, 4]
    aiff_bytes = _build_aiff_or_aifc(samples, channels=1, sample_width=2, frame_rate=22050)
    monkeypatch.setitem(
        sys.modules, "pyttsx3", _fake_pyttsx3(silent_names=set(), write_bytes=aiff_bytes)
    )

    sleep_calls = {"n": 0}

    def counting_sleep(_seconds: float) -> None:
        sleep_calls["n"] += 1

    tick = {"t": 0.0}

    def fake_monotonic() -> float:
        tick["t"] += synthetic_audio.STABILIZE_POLL_SECONDS
        return tick["t"]

    paths = synthetic_audio.generate(
        tmp_path, only={"standup.wav"}, sleep=counting_sleep, monotonic=fake_monotonic
    )

    assert sleep_calls["n"] == synthetic_audio.STABILIZE_REQUIRED_STABLE_READS - 1
    rate, data = wavfile.read(paths["standup.wav"])
    assert data.tolist() == samples


# =============================================================================
# normalize_speech_audio_to_wav - pure conversion, no TTS engine involved
# =============================================================================


def test_normalize_leaves_a_real_wav_file_byte_identical() -> None:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(22050)
        writer.writeframes(struct.pack("<4h", 1, 2, 3, 4))
    wav_bytes = buffer.getvalue()

    result = synthetic_audio.normalize_speech_audio_to_wav(wav_bytes, source_name="x.wav")
    assert result == wav_bytes


def test_normalize_converts_aiff_to_a_readable_wav_with_correct_samples() -> None:
    """A wrong byte order would change these exact values (e.g. 0x0100 vs 0x0001),
    so this test can actually fail if the big-endian -> little-endian conversion
    is wrong, not just if the file becomes unreadable."""
    samples = [1, 256, -1, -256, 12345, -12345]
    aiff_bytes = _build_aiff_or_aifc(samples, channels=1, sample_width=2, frame_rate=16000)

    wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(aiff_bytes, source_name="a.wav")

    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16000
        raw = reader.readframes(reader.getnframes())
    assert list(struct.unpack(f"<{len(samples)}h", raw)) == samples


def test_normalize_converts_aifc_sowt_to_a_readable_wav_with_correct_samples() -> None:
    """'sowt' AIFF-C data is already little-endian; if the converter wrongly
    byte-swapped it anyway, these sample values would come out scrambled."""
    samples = [1, 256, -1, -256, 12345, -12345]
    aifc_bytes = _build_aiff_or_aifc(
        samples, channels=1, sample_width=2, frame_rate=44100, aifc_compression=b"sowt"
    )

    wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(aifc_bytes, source_name="b.wav")

    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 44100
        raw = reader.readframes(reader.getnframes())
    assert list(struct.unpack(f"<{len(samples)}h", raw)) == samples


def test_normalize_stereo_aiff_round_trips_with_scipy() -> None:
    """Exercises the real downstream reader (assistant.py uses scipy.io.wavfile),
    with two channels, to prove channel count survives the conversion too."""
    # interleaved L/R: (L0,R0,L1,R1)
    samples = [1000, -1000, 2000, -2000]
    aiff_bytes = _build_aiff_or_aifc(samples, channels=2, sample_width=2, frame_rate=22050)

    wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(aiff_bytes, source_name="c.wav")

    rate, data = wavfile.read(BytesIO(wav_bytes))
    assert rate == 22050
    assert data.shape == (2, 2)
    assert data.tolist() == [[1000, -1000], [2000, -2000]]


def _wav_frames(wav_bytes: bytes) -> tuple[int, bytes]:
    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        return reader.getsampwidth(), reader.readframes(reader.getnframes())


def test_normalize_converts_8_bit_signed_aiff_to_unsigned_wav() -> None:
    """AIFF 8-bit PCM is signed and WAV 8-bit PCM is unsigned: -128 must
    become 0, 0 must become 128 and 127 must become 255."""
    aiff_bytes = _build_aiff_or_aifc([-128, -1, 0, 127], sample_width=1)

    width, frames = _wav_frames(
        synthetic_audio.normalize_speech_audio_to_wav(aiff_bytes, source_name="8bit.wav")
    )

    assert width == 1
    assert list(frames) == [0, 127, 128, 255]


def test_normalize_converts_24_bit_big_endian_aiff_to_little_endian_wav() -> None:
    samples = [1, -2, 0x123456, -0x123456]
    aiff_bytes = _build_aiff_or_aifc(samples, sample_width=3)

    width, frames = _wav_frames(
        synthetic_audio.normalize_speech_audio_to_wav(aiff_bytes, source_name="24bit.wav")
    )

    assert width == 3
    assert frames == b"".join((s & 0xFFFFFF).to_bytes(3, "little") for s in samples)


def test_normalize_converts_32_bit_big_endian_aiff_to_little_endian_wav() -> None:
    samples = [1, -2, 0x12345678, -0x12345678]
    aiff_bytes = _build_aiff_or_aifc(samples, sample_width=4)

    width, frames = _wav_frames(
        synthetic_audio.normalize_speech_audio_to_wav(aiff_bytes, source_name="32bit.wav")
    )

    assert width == 4
    assert frames == struct.pack("<4l", *samples)


def test_normalize_drops_a_trailing_partial_frame() -> None:
    """A truncated SSND chunk can end mid-frame. Here 'sowt' 16-bit mono data
    declares 3 frames but carries 5 bytes; the WAV must hold the 2 whole
    frames only, with a data chunk that describes exactly those 4 bytes
    ('sowt' needs no byte swap, so nothing else would drop the odd byte)."""
    comm_body = struct.pack(">hlh", 1, 3, 16) + _write_ieee_extended(22050.0) + b"sowt\x00"
    ssnd_body = struct.pack(">II", 0, 0) + struct.pack("<2h", 1, 2) + b"\x03"
    body = b"AIFC" + _chunk(b"COMM", comm_body) + _chunk(b"SSND", ssnd_body)
    aifc_bytes = b"FORM" + struct.pack(">I", len(body)) + body

    wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(aifc_bytes, source_name="t.wav")

    assert len(wav_bytes) == 44 + 4  # canonical 44-byte header + two 2-byte frames
    assert _wav_frames(wav_bytes) == (2, struct.pack("<2h", 1, 2))


def test_normalize_rejects_an_unknown_header() -> None:
    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="not-a-file"):
        synthetic_audio.normalize_speech_audio_to_wav(
            b"this is not any known audio header at all", source_name="not-a-file.wav"
        )


def test_normalize_rejects_a_compressed_aifc_type() -> None:
    aifc_bytes = _build_aiff_or_aifc(
        [1, 2, 3, 4], channels=1, sample_width=2, frame_rate=22050, aifc_compression=b"ima4"
    )
    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="compressed.wav"):
        synthetic_audio.normalize_speech_audio_to_wav(aifc_bytes, source_name="compressed.wav")


def test_normalize_rejects_an_aifc_comm_chunk_without_a_compression_type() -> None:
    """An AIFF-C COMM chunk of 18-21 bytes has no compression type; it must
    be rejected, not silently read as uncompressed big-endian PCM."""
    comm_body = struct.pack(">hlh", 1, 2, 16) + _write_ieee_extended(22050.0)  # 18 bytes
    ssnd_body = struct.pack(">II", 0, 0) + struct.pack(">2h", 1, 2)
    body = b"AIFC" + _chunk(b"COMM", comm_body) + _chunk(b"SSND", ssnd_body)
    aifc_bytes = b"FORM" + struct.pack(">I", len(body)) + body

    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="compression type"):
        synthetic_audio.normalize_speech_audio_to_wav(aifc_bytes, source_name="short-comm.wav")


def test_normalize_rejects_malformed_chunks() -> None:
    # A FORM/AIFF header whose only chunk declares a size larger than the
    # remaining bytes - the kind of truncated/corrupt file _iter_chunks must
    # reject cleanly rather than reading past the end.
    truncated = b"FORM" + struct.pack(">I", 100) + b"AIFF" + b"COMM" + struct.pack(">I", 9999)
    with pytest.raises(synthetic_audio.TTSEngineUnavailableError, match="broken.wav"):
        synthetic_audio.normalize_speech_audio_to_wav(truncated, source_name="broken.wav")


# =============================================================================
# Byte-order check - diagnostic only, the header's order is always used
# =============================================================================
#
# These use a smooth, audio-like signal (not a handful of arbitrary integers)
# so the check has something realistic to score, and white noise for the
# "cannot tell" case.


def _build_test_signal(num_samples: int = 11025, sample_rate: int = 22050) -> list[int]:
    """A 440 Hz tone under a slower 2 Hz amplitude envelope, ~0.5s at
    22050Hz by default - smooth, speech-like data, so the byte-order check
    has an actual signal to tell apart from noise."""
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
        value = envelope * 20000 * math.sin(2 * math.pi * 440.0 * t)
        samples.append(int(value))
    return samples


def _build_white_noise_signal(num_samples: int = 11025, seed: int = 12345) -> list[int]:
    """White noise (16-bit range, seeded for reproducibility) - by
    construction has no more structure decoded one way than the other, so
    the check should find neither order convincingly smoother."""
    rng = random.Random(seed)
    return [rng.randint(-32000, 32000) for _ in range(num_samples)]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    # caplog.at_level(..., logger=...) captures from this module's own logger
    # directly, whatever its propagate setting.
    return [
        r
        for r in caplog.records
        if r.name == synthetic_audio.log.name and r.levelno == logging.WARNING
    ]


def test_normalize_correctly_labelled_twos_round_trips_with_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Big-endian speech-like data labelled 'twos' (big-endian): the samples
    round-trip exactly and nothing is logged at WARNING."""
    samples = _build_test_signal()
    aifc_bytes = _build_aiff_or_aifc(samples, aifc_compression=b"twos", actual_byte_order=">")

    with caplog.at_level(logging.WARNING, logger=synthetic_audio.log.name):
        wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(aifc_bytes, source_name="ok.wav")

    _, raw = _wav_frames(wav_bytes)
    assert list(struct.unpack(f"<{len(samples)}h", raw)) == samples
    assert _warnings(caplog) == []


def test_normalize_logs_a_disagreement_but_uses_the_label(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Little-endian data labelled 'twos' (big-endian). The header's order is
    used, so every sample is byte-swapped from what was written, and one
    WARNING names the file and both orders. A converter that followed the
    signal instead would round-trip the samples and fail the first assertion."""
    samples = _build_test_signal()
    aifc_bytes = _build_aiff_or_aifc(samples, aifc_compression=b"twos", actual_byte_order="<")

    with caplog.at_level(logging.WARNING, logger=synthetic_audio.log.name):
        wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(
            aifc_bytes, source_name="mislabelled.wav"
        )

    _, raw = _wav_frames(wav_bytes)
    written = struct.pack(f"<{len(samples)}h", *samples)
    label_decoding = list(struct.unpack(f">{len(samples)}h", written))
    assert list(struct.unpack(f"<{len(samples)}h", raw)) == label_decoding
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "mislabelled.wav" in message
    assert "looks little-endian" in message
    assert "header says big-endian" in message


def test_normalize_correctly_labelled_sowt_round_trips_with_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    samples = _build_test_signal()
    aifc_bytes = _build_aiff_or_aifc(samples, aifc_compression=b"sowt", actual_byte_order="<")

    with caplog.at_level(logging.WARNING, logger=synthetic_audio.log.name):
        wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(
            aifc_bytes, source_name="sowt.wav"
        )

    _, raw = _wav_frames(wav_bytes)
    assert list(struct.unpack(f"<{len(samples)}h", raw)) == samples
    assert _warnings(caplog) == []


def test_normalize_white_noise_uses_the_label_with_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """White noise gives an inconclusive check: the label is used and nothing
    is logged at WARNING."""
    samples = _build_white_noise_signal()
    aifc_bytes = _build_aiff_or_aifc(samples, aifc_compression=b"twos", actual_byte_order=">")

    with caplog.at_level(logging.WARNING, logger=synthetic_audio.log.name):
        wav_bytes = synthetic_audio.normalize_speech_audio_to_wav(
            aifc_bytes, source_name="noise.wav"
        )

    _, raw = _wav_frames(wav_bytes)
    assert list(struct.unpack(f"<{len(samples)}h", raw)) == samples
    assert _warnings(caplog) == []


def test_check_byte_order_agrees_with_a_correct_label() -> None:
    samples = _build_test_signal(num_samples=4000)
    big_pcm = struct.pack(f">{len(samples)}h", *samples)

    check = synthetic_audio._check_byte_order(
        big_pcm, sample_width=2, channels=1, sample_rate=22050, label_big_endian=True
    )

    assert check.detected_big_endian is True
    assert check.disagrees_with_label is False
    assert check.big_endian_score < check.little_endian_score


def test_check_byte_order_reports_the_smoother_decoding_against_a_wrong_label() -> None:
    samples = _build_test_signal(num_samples=4000)
    big_pcm = struct.pack(f">{len(samples)}h", *samples)

    check = synthetic_audio._check_byte_order(
        big_pcm, sample_width=2, channels=1, sample_rate=22050, label_big_endian=False
    )

    assert check.detected_big_endian is True
    assert check.label_big_endian is False
    assert check.disagrees_with_label is True


def test_check_byte_order_is_inconclusive_for_white_noise() -> None:
    samples = _build_white_noise_signal(num_samples=4000)
    pcm = struct.pack(f">{len(samples)}h", *samples)

    check = synthetic_audio._check_byte_order(
        pcm, sample_width=2, channels=1, sample_rate=22050, label_big_endian=True
    )

    assert check.detected_big_endian is None
    assert check.disagrees_with_label is False


def test_smoothness_score_is_lower_for_the_smooth_signal_than_for_noise() -> None:
    """A wrong-byte-order decoding of real audio should score close to (or
    worse than) genuine white noise -- if it scored much *better* than
    noise, the smoothness heuristic itself would be backwards."""
    signal = np.array(_build_test_signal(num_samples=4000), dtype=np.int64)
    noise = np.array(_build_white_noise_signal(num_samples=4000), dtype=np.int64)

    signal_score = synthetic_audio._smoothness_score(signal)
    noise_score = synthetic_audio._smoothness_score(noise)

    assert signal_score < noise_score


# =============================================================================
# extract_audio_facts - diagnostics
# =============================================================================
#
# These facts make a bad file visible in the log: the source
# container/compression, channel/width/rate, and declared vs. actually-present
# frame count, which tells a conversion problem apart from a truncated write.


def _build_wav(samples: list[int], *, channels: int = 1, sample_rate: int = 22050) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


def test_extract_audio_facts_on_a_riff_wave_file() -> None:
    """The Linux/Windows baseline: real RIFF/WAVE bytes, never touched by
    the AIFF conversion path at all."""
    samples = [1, 2, 3, 4, 5, 6]
    wav_bytes = _build_wav(samples, channels=1, sample_rate=22050)

    facts = synthetic_audio.extract_audio_facts(wav_bytes)

    assert facts.container == "RIFF"
    assert facts.compression == "PCM"
    assert facts.channels == 1
    assert facts.sample_width == 2
    assert facts.sample_rate == 22050
    assert facts.declared_frames == len(samples)
    assert facts.actual_frames == len(samples)
    assert facts.duration_seconds == pytest.approx(len(samples) / 22050)
    assert facts.file_size_bytes == len(wav_bytes)
    assert facts.note == ""


def test_extract_audio_facts_on_an_aiff_file() -> None:
    samples = [10, 20, 30, 40]
    aiff_bytes = _build_aiff_or_aifc(samples, channels=1, sample_width=2, frame_rate=16000)

    facts = synthetic_audio.extract_audio_facts(aiff_bytes)

    assert facts.container == "AIFF"
    assert facts.compression == "NONE"
    assert facts.channels == 1
    assert facts.sample_width == 2
    assert facts.sample_rate == 16000
    assert facts.declared_frames == len(samples)
    assert facts.actual_frames == len(samples)
    assert facts.file_size_bytes == len(aiff_bytes)
    assert facts.note == ""


def test_extract_audio_facts_on_an_aifc_sowt_file() -> None:
    samples = [10, 20, 30, 40]
    aifc_bytes = _build_aiff_or_aifc(
        samples, channels=1, sample_width=2, frame_rate=44100, aifc_compression=b"sowt"
    )

    facts = synthetic_audio.extract_audio_facts(aifc_bytes)

    assert facts.container == "AIFC"
    assert facts.compression == "sowt"
    assert facts.sample_rate == 44100
    assert facts.declared_frames == len(samples) == facts.actual_frames


def test_extract_audio_facts_detects_a_truncated_ssnd_chunk() -> None:
    """Models a write read before it finished: fewer sample-data bytes on
    disk than ``COMM``'s frame count declares. A
    wrong ``actual_frames`` computation (e.g. trusting the declared size
    instead of the bytes actually present) would make this mismatch vanish
    and this test would fail."""
    samples = list(range(100))  # 100 frames, 16-bit mono
    full = _build_aiff_or_aifc(samples, channels=1, sample_width=2, frame_rate=22050)
    truncated = full[:-50]  # chop 50 bytes off the tail: an incomplete write

    facts = synthetic_audio.extract_audio_facts(truncated)

    assert facts.container == "AIFF"
    assert facts.declared_frames == 100
    assert facts.actual_frames == 75  # (200 pcm bytes - 50 chopped) / 2 bytes/frame
    assert facts.declared_frames != facts.actual_frames
    assert facts.duration_seconds == pytest.approx(75 / 22050)


def test_extract_audio_facts_on_unrecognized_bytes_never_raises() -> None:
    facts = synthetic_audio.extract_audio_facts(b"nonsense, not any audio header")
    assert facts.container == "unknown"
    assert facts.note != ""


def test_audio_facts_format_flags_a_frame_count_mismatch() -> None:
    """The line fed into DemoResult notes and log output must actually show
    a mismatch when there is one, and stay silent about it when there
    isn't - and never contain a newline, since it is meant to sit on one
    log/table line."""
    matching = synthetic_audio.AudioFacts("RIFF", "PCM", 1, 2, 22050, 10, 10, 10 / 22050, 100)
    line = matching.format("ok.wav")
    assert "MISMATCH" not in line
    assert "\n" not in line

    mismatched = synthetic_audio.AudioFacts("AIFF", "NONE", 1, 2, 22050, 100, 75, 75 / 22050, 300)
    line = mismatched.format("bad.wav")
    assert "MISMATCH" in line
    assert "declared=100" in line
    assert "actual=75" in line
    assert "\n" not in line
