"""synthetic_audio.py — synthesize short meeting recordings for testing (offline).

Project P10. Uses pyttsx3 (Windows SAPI5 / eSpeak on Linux) to turn two short
meeting scripts — each with clear topics, decisions and action items — into
WAV files under ``./data``. This keeps the project self-contained: no
copyrighted or expiring demo downloads, and the transcript content is known
ahead of time so the Whisper -> LLM chain can be checked end to end.

Run
    uv run python -m projects.p10_meeting_assistant.synthetic_audio
"""

from __future__ import annotations

import logging
import platform
import struct
import sys
import time
import wave
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np  # numpy is in the "core" dependency group, always available here

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

log = logging.getLogger(__name__)


class TTSEngineUnavailableError(RuntimeError):
    """Raised when the OS text-to-speech engine ``pyttsx3`` depends on is missing.

    ``pyttsx3`` wraps a per-OS driver (SAPI5 on Windows, ``espeak``/
    ``espeak-ng`` on Linux, ``NSSpeechSynthesizer`` on macOS) rather than
    shipping its own; on Linux that system package is often absent (e.g. a
    bare CI runner), and ``pyttsx3.init()`` raises. Callers — notably
    ``demo()`` — catch this specifically to degrade to a ``skipped`` result
    with an actionable message, instead of a raw traceback or a ``failed``
    status.
    """


# =============================================================================
# macOS AIFF -> WAV normalization
# =============================================================================
#
# On macOS, pyttsx3's driver (NSSpeechSynthesizer) writes AIFF or AIFF-C
# bytes to the path it is given, regardless of the ``.wav`` extension
# ``generate()`` asks for. ``assistant.load_audio`` reads with
# ``scipy.io.wavfile.read``, which only accepts RIFF/RIFX/RF64 — so a
# mislabelled AIFF file fails two steps removed from the real cause
# (``ValueError: ... File format b'FORM' not understood``) deep inside
# Whisper's loader. The functions below detect that case and convert the
# file to a standard RIFF PCM WAV in place, immediately after synthesis.
#
# Python 3.13 removed the stdlib ``aifc`` module entirely (PEP 594), so AIFF
# is parsed here by hand with ``struct`` — chunk sizes are big-endian, odd
# chunk bodies are padded to an even length, and the sample rate in ``COMM``
# is a big-endian 80-bit IEEE-754 extended-precision float (the decoder below
# ports the same algorithm ``aifc._read_float`` used).


def _read_ieee_extended(raw: bytes) -> float:
    """Decode a big-endian 80-bit IEEE-754 extended-precision float, the
    encoding AIFF's ``COMM`` chunk uses for the sample rate."""
    if len(raw) != 10:
        raise ValueError(f"expected 10 bytes for an IEEE-extended float, got {len(raw)}")
    expon = int.from_bytes(raw[0:2], "big")
    sign = 1
    if expon & 0x8000:
        sign = -1
        expon &= 0x7FFF
    himant = int.from_bytes(raw[2:6], "big")
    lomant = int.from_bytes(raw[6:10], "big")
    if expon == 0 and himant == 0 and lomant == 0:
        return 0.0
    if expon == 0x7FFF:
        raise ValueError("IEEE-extended float is infinite or NaN")
    expon -= 16383
    value = (himant * 4294967296 + lomant) * (2.0 ** (expon - 63))
    return sign * value


def _iter_chunks(data: bytes, start: int) -> Iterator[tuple[str, bytes]]:
    """Yield ``(chunk_id, body)`` for each top-level chunk in an IFF-style
    container (AIFF/AIFF-C), starting at byte offset ``start`` (right after
    the ``FORM``/size/form-type header). Chunk sizes are big-endian; an odd
    body is followed by one pad byte that is skipped, not yielded."""
    pos = start
    end = len(data)
    while pos + 8 <= end:
        chunk_id = data[pos : pos + 4].decode("ascii", errors="replace")
        size = int.from_bytes(data[pos + 4 : pos + 8], "big")
        body_start = pos + 8
        body_end = body_start + size
        if body_end > end:
            raise ValueError(f"chunk '{chunk_id}' declares size {size} past the end of the file")
        yield chunk_id, data[body_start:body_end]
        pos = body_end + (size & 1)  # chunks are padded to an even length


def _byteswap(data: bytes, sample_width: int) -> bytes:
    """Reverse the byte order of every ``sample_width``-byte sample in
    ``data`` (big-endian <-> little-endian). A no-op for 1-byte samples."""
    if sample_width <= 1:
        return data
    n = (len(data) // sample_width) * sample_width
    out = bytearray(n)
    for i in range(0, n, sample_width):
        out[i : i + sample_width] = data[i : i + sample_width][::-1]
    return bytes(out)


# =============================================================================
# Byte-order detection from the signal (fix round 3)
# =============================================================================
#
# heavy.yml run 35084927134: round 2's stabilization wait worked — the
# macOS file was complete (608937 declared == actual frames, 27.62s for a
# 73-word script, exactly right) — but the transcript was still noise
# ('il Tottenham ca cagggg...', 4/73 words). normalize_speech_audio_to_wav
# implements AIFF-C's documented semantics correctly ('twos' = big-endian,
# so it byte-swaps); Windows exercises the identical downstream path and
# transcribes 73/73. The leading suspect: Apple's driver labels the file
# 'twos' but actually writes little-endian samples — but that is an
# inference about a machine this code cannot touch, so the decision is
# made from the signal itself rather than hard-coded.


def _decode_pcm_samples(pcm: bytes, sample_width: int, *, big_endian: bool) -> np.ndarray:
    """Decode raw signed PCM bytes as a 1-D array of sample values (one
    entry per sample per channel — channels are not de-interleaved, which
    smoothness scoring does not need). Supports the sample widths AIFF/
    AIFF-C can declare: 1 (8-bit, always signed at this point — no byte
    order to decode), 2, 3 (24-bit, assembled by hand since neither numpy
    nor ``struct`` has a native int24 type) and 4 bytes."""
    n = len(pcm) // sample_width
    trimmed = pcm[: n * sample_width]
    if n == 0:
        return np.array([], dtype=np.int64)
    if sample_width == 1:
        return np.frombuffer(trimmed, dtype=np.int8).astype(np.int64)
    if sample_width == 2:
        dtype = ">i2" if big_endian else "<i2"
        return np.frombuffer(trimmed, dtype=dtype).astype(np.int64)
    if sample_width == 4:
        dtype = ">i4" if big_endian else "<i4"
        return np.frombuffer(trimmed, dtype=dtype).astype(np.int64)
    if sample_width == 3:
        triples = np.frombuffer(trimmed, dtype=np.uint8).reshape(n, 3).astype(np.int64)
        hi, mid, lo = (0, 1, 2) if big_endian else (2, 1, 0)
        value = (triples[:, hi] << 16) | (triples[:, mid] << 8) | triples[:, lo]
        return np.where(value & 0x800000, value - 0x1000000, value)  # sign-extend 24-bit
    raise ValueError(f"unsupported sample width for PCM decoding: {sample_width}")


def _smoothness_score(samples: np.ndarray) -> float:
    """Lower is smoother: mean absolute sample-to-sample difference, over
    mean absolute sample value. Correctly-ordered speech at 22 kHz is a
    smooth waveform relative to its own amplitude (consecutive samples are
    close together); decoded under the wrong byte order, a PCM stream is
    close to white noise (consecutive samples are essentially
    uncorrelated), so the ratio between the two scores tends to be large.
    """
    if samples.size < 2:
        return float("inf")
    values = samples.astype(np.float64)
    mean_abs_value = float(np.mean(np.abs(values)))
    if mean_abs_value == 0.0:
        return float("inf")  # silence — no signal to judge smoothness from
    mean_abs_diff = float(np.mean(np.abs(np.diff(values))))
    return mean_abs_diff / mean_abs_value


def _amplitude_stats(samples: np.ndarray, sample_width: int) -> tuple[float, float, float]:
    """RMS, peak and clipped-sample fraction of ``samples``, each expressed
    as a fraction of full scale for a signed ``sample_width``-byte sample
    (e.g. full scale 32768 for 16-bit)."""
    if samples.size == 0:
        return 0.0, 0.0, 0.0
    full_scale = float(2 ** (sample_width * 8 - 1))
    values = samples.astype(np.float64)
    rms = float(np.sqrt(np.mean(values * values))) / full_scale
    peak = float(np.max(np.abs(values))) / full_scale
    clip_threshold = full_scale - 1  # the largest representable magnitude
    clipped = float(np.mean(np.abs(values) >= clip_threshold))
    return rms, peak, clipped


# Named margin (fix round 3): the worse (noisier) smoothness score must be
# at least this many times larger than the better one before detection is
# trusted over the header's own label — otherwise (near-silence, very short
# clips, tones that happen to look similar either way) the label's order is
# kept and the result is reported as inconclusive rather than guessed at.
BYTE_ORDER_DETECTION_MARGIN = 1.5
# How much of the signal to decode/score for detection — a bounded prefix
# is plenty to tell smooth speech from noise, and scoring a multi-second
# recording in full would be needless work.
BYTE_ORDER_DETECTION_MAX_SECONDS = 5.0


@dataclass(frozen=True)
class ByteOrderDecision:
    """The outcome of deciding 16/24/32-bit PCM byte order from the signal
    rather than trusting AIFF-C's compression-type label."""

    big_endian: bool  # the order to actually use
    label_big_endian: bool  # what the header/compression tag said
    big_endian_score: float  # smoothness decoding as big-endian (lower = smoother)
    little_endian_score: float  # smoothness decoding as little-endian
    inconclusive: bool  # True: neither order was convincingly smoother; kept the label

    @property
    def overrides_label(self) -> bool:
        return not self.inconclusive and self.big_endian != self.label_big_endian


def _decide_byte_order(
    pcm: bytes,
    *,
    sample_width: int,
    channels: int,
    sample_rate: int,
    label_big_endian: bool,
) -> ByteOrderDecision:
    """Decide whether ``pcm`` (16/24/32-bit signed PCM) is actually
    big-endian or little-endian by scoring both decodings' smoothness over
    a bounded prefix (``BYTE_ORDER_DETECTION_MAX_SECONDS``) and picking
    whichever is at least ``BYTE_ORDER_DETECTION_MARGIN`` times smoother
    than the other. If neither is convincingly smoother, falls back to
    ``label_big_endian`` (``inconclusive=True``) rather than guessing. A
    small, pure function — no I/O, no logging — so it is directly testable.
    """
    max_bytes = int(sample_rate * channels * sample_width * BYTE_ORDER_DETECTION_MAX_SECONDS)
    prefix = pcm[:max_bytes] if max_bytes > 0 else pcm
    big_score = _smoothness_score(_decode_pcm_samples(prefix, sample_width, big_endian=True))
    little_score = _smoothness_score(_decode_pcm_samples(prefix, sample_width, big_endian=False))

    better_big = big_score <= little_score
    better_score = big_score if better_big else little_score
    worse_score = little_score if better_big else big_score
    convincing = (
        worse_score > 0
        if better_score == 0
        else worse_score >= better_score * BYTE_ORDER_DETECTION_MARGIN
    )

    if convincing:
        return ByteOrderDecision(better_big, label_big_endian, big_score, little_score, False)
    return ByteOrderDecision(label_big_endian, label_big_endian, big_score, little_score, True)


def _order_name(big_endian: bool) -> str:
    return "big" if big_endian else "little"


def _log_byte_order_decision(source_name: str, decision: ByteOrderDecision) -> None:
    """Log detection's outcome — silent when detection is conclusive and
    agrees with the label (the common, unremarkable case); ``INFO`` when
    inconclusive (falling back to the label); ``WARNING`` naming the label,
    both scores and the order actually used when detection overrides the
    label (fix round 3's core requirement — this is what would have caught
    the 'twos'-labelled-but-little-endian macOS files directly in the log).
    """
    if decision.inconclusive:
        log.info(
            "%s: byte-order detection inconclusive (big=%.4f little=%.4f, margin=%.1fx "
            "required); using the label's order (%s)",
            source_name,
            decision.big_endian_score,
            decision.little_endian_score,
            BYTE_ORDER_DETECTION_MARGIN,
            _order_name(decision.label_big_endian),
        )
    elif decision.overrides_label:
        log.warning(
            "%s: detected byte order (%s) disagrees with the AIFF-C label (%s); using the "
            "detected order (scores: big=%.4f little=%.4f)",
            source_name,
            _order_name(decision.big_endian),
            _order_name(decision.label_big_endian),
            decision.big_endian_score,
            decision.little_endian_score,
        )


def normalize_speech_audio_to_wav(raw: bytes, *, source_name: str) -> bytes:
    """Pure conversion: AIFF/AIFF-C bytes -> standard RIFF PCM WAV bytes.

    Takes and returns raw bytes only (no file I/O), so it is testable with no
    TTS engine, no OS driver, and no fixture files.

    - ``RIFF....WAVE`` -> returned byte-identical (already a standard WAV),
      no byte-order detection performed.
    - ``FORM....AIFF`` (always uncompressed PCM) or ``FORM....AIFC`` with an
      uncompressed PCM compression type (``NONE``/``twos``/``sowt``) ->
      converted to a standard little-endian RIFF PCM WAV with the same
      channel count, sample width and sample rate. AIFF's 8-bit PCM is
      signed; WAV's 8-bit PCM is unsigned, so 8-bit samples are shifted by
      128 during conversion (no byte order to decide for single-byte
      samples). For wider samples (16/24/32-bit), fix round 3: real macOS
      output has been observed to label little-endian samples ``twos``
      (which is documented to mean big-endian) — see
      ``_decide_byte_order``'s module comment — so the compression tag is
      treated only as a *label*, and the byte order actually used is
      decided from the signal itself (``_decide_byte_order``), falling back
      to the label when detection is inconclusive. A detected order that
      disagrees with the label is logged as a ``WARNING``; an inconclusive
      detection is logged at ``INFO``.
    - Anything else (unrecognized header, a compressed AIFC type, or
      malformed/truncated chunks) raises ``TTSEngineUnavailableError`` naming
      ``source_name`` and what was found, rather than handing a mislabelled
      file downstream — the same contract ``generate()`` already has for a
      missing TTS engine.
    """
    if len(raw) < 12:
        raise TTSEngineUnavailableError(
            f"'{source_name}' is only {len(raw)} bytes — too short to be a WAV or AIFF file "
            "pyttsx3 could plausibly have written."
        )

    riff_id, form_type = raw[0:4], raw[8:12]

    if riff_id == b"RIFF" and form_type == b"WAVE":
        return raw  # already a standard WAV — left untouched, byte-identical

    if riff_id != b"FORM" or form_type not in (b"AIFF", b"AIFC"):
        raise TTSEngineUnavailableError(
            f"'{source_name}' is neither a RIFF/WAVE nor a FORM/AIFF(-C) file "
            f"(saw {riff_id!r} / {form_type!r}); refusing to hand a file of unknown "
            "format downstream."
        )

    comm: dict[str, int | float] | None = None
    compression_type = b"NONE"
    ssnd_data: bytes | None = None
    try:
        for chunk_id, body in _iter_chunks(raw, 12):
            if chunk_id == "COMM":
                if len(body) < 18:
                    raise ValueError("COMM chunk is shorter than the required 18 bytes")
                channels, frames, sample_size = struct.unpack(">hlh", body[0:8])
                sample_rate = _read_ieee_extended(body[8:18])
                comm = {
                    "channels": channels,
                    "frames": frames,
                    "sample_size": sample_size,
                    "sample_rate": sample_rate,
                }
                if form_type == b"AIFC" and len(body) >= 22:
                    compression_type = body[18:22]
            elif chunk_id == "SSND":
                if len(body) < 8:
                    raise ValueError("SSND chunk is shorter than the required 8 bytes")
                offset, _block_size = struct.unpack(">LL", body[0:8])
                ssnd_data = body[8 + offset :]
    except (struct.error, ValueError) as exc:
        raise TTSEngineUnavailableError(
            f"'{source_name}' looked like an AIFF/AIFF-C file but its chunks were malformed: {exc}"
        ) from exc

    if comm is None or ssnd_data is None:
        missing = "COMM" if comm is None else "SSND"
        raise TTSEngineUnavailableError(
            f"'{source_name}' is missing its required {missing} chunk; cannot convert it to WAV."
        )

    if form_type == b"AIFF" or compression_type in (b"NONE", b"none", b"twos", b"TWOS"):
        label_big_endian = True
    elif compression_type in (b"sowt", b"SOWT"):
        label_big_endian = False
    else:
        raise TTSEngineUnavailableError(
            f"'{source_name}' is a compressed AIFF-C file (compression type "
            f"{compression_type!r}); only uncompressed PCM ('NONE'/'twos'/'sowt') is supported."
        )

    channels = int(comm["channels"])
    sample_size = int(comm["sample_size"])
    sample_rate = round(comm["sample_rate"])
    if sample_size <= 0 or sample_size % 8 != 0:
        raise TTSEngineUnavailableError(
            f"'{source_name}' uses a {sample_size}-bit sample size, which is not a whole "
            "number of bytes; cannot convert it to WAV."
        )
    sample_width = sample_size // 8

    frame_count = int(comm["frames"])
    expected_len = frame_count * channels * sample_width
    pcm = ssnd_data[:expected_len] if len(ssnd_data) >= expected_len else ssnd_data

    if sample_width == 1:
        # AIFF 8-bit PCM is signed; WAV 8-bit PCM is unsigned — shift by 128.
        pcm = bytes((byte + 128) & 0xFF for byte in pcm)
    else:
        decision = _decide_byte_order(
            pcm,
            sample_width=sample_width,
            channels=channels,
            sample_rate=sample_rate,
            label_big_endian=label_big_endian,
        )
        _log_byte_order_decision(source_name, decision)
        if decision.big_endian:
            pcm = _byteswap(pcm, sample_width)
        # else: decided little-endian — no swap needed.

    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


# =============================================================================
# Audio facts (fix round 1) — diagnose "the demo said ok but Whisper heard
# almost nothing"
# =============================================================================
#
# heavy.yml's macOS run went green (the crash from round 0 is gone) but
# transcribed only 4 words of a 73-word script — a silent failure, not a
# crash. Two hypotheses: (a) the AIFF/AIFF-C -> WAV conversion above
# mis-handles some real-file property these hand-built fixtures don't cover,
# or (b) NSSpeechSynthesizer's ``runAndWait()`` returns before the file is
# completely flushed, leaving a truncated ``SSND`` chunk that
# ``normalize_speech_audio_to_wav`` silently trims to whatever bytes are
# actually there (see its ``pcm = ssnd_data[:expected_len] if len(ssnd_data)
# >= expected_len else ssnd_data`` line) instead of raising. Telling these
# apart needs facts from the *source* bytes exactly as pyttsx3 wrote them,
# before any conversion/trimming touches them — captured here and logged
# for every generated file (RIFF included, as a Linux/Windows baseline).


@dataclass(frozen=True)
class AudioFacts:
    """Best-effort diagnostic snapshot of one audio file's header, read from
    its raw bytes exactly as written to disk (i.e. captured *before*
    ``normalize_speech_audio_to_wav`` may trim a short ``SSND``/``data``
    chunk to fit — the whole point is to see the mismatch that conversion
    would otherwise hide)."""

    container: str  # "RIFF", "AIFF", "AIFC", or "unknown"
    compression: str  # "PCM" / a WAV format code / an AIFF-C compression tag
    channels: int
    sample_width: int  # bytes per sample; 0 if unknown
    sample_rate: int
    declared_frames: int  # frame count the header claims
    actual_frames: int  # frame count the sample-data bytes actually cover
    duration_seconds: float  # actual_frames / sample_rate — the *real* length
    file_size_bytes: int
    note: str = ""  # non-empty only when parsing hit something unexpected
    # Fix round 3: byte-order detection + amplitude diagnostics, populated
    # only for AIFF/AIFF-C sources with sample_width in (2, 3, 4) — RIFF/WAV
    # and 8-bit sources have no byte-order ambiguity to detect, so these
    # stay at their "n/a"/None defaults for them.
    byte_order_used: str = "n/a"  # "big", "little", or "n/a"
    byte_order_label: str = "n/a"  # what the header/compression tag said
    byte_order_big_score: float | None = None  # lower = smoother
    byte_order_little_score: float | None = None
    byte_order_inconclusive: bool = False
    rms: float | None = None  # of the chosen decoding, as a fraction of full scale
    peak: float | None = None  # ditto
    clipped_fraction: float | None = None  # ditto

    def format(self, name: str) -> str:
        """One single-line, human-readable summary — safe to drop straight
        into a log line or a ``DemoResult`` note (no embedded newlines)."""
        mismatch = (
            f" FRAME-COUNT-MISMATCH(declared={self.declared_frames}, actual={self.actual_frames})"
            if self.declared_frames != self.actual_frames
            else ""
        )
        note = f" note={self.note!r}" if self.note else ""
        byte_order = ""
        if self.byte_order_used != "n/a":
            flag = (
                " inconclusive"
                if self.byte_order_inconclusive
                else (
                    f" OVERRIDES-LABEL({self.byte_order_label})"
                    if self.byte_order_used != self.byte_order_label
                    else ""
                )
            )
            byte_order = (
                f" byte_order={self.byte_order_used}{flag} "
                f"(big={self.byte_order_big_score:.4f} little={self.byte_order_little_score:.4f}) "
                f"rms={self.rms:.4f} peak={self.peak:.4f} clipped={self.clipped_fraction:.4%}"
            )
        return (
            f"{name}: container={self.container} compression={self.compression} "
            f"channels={self.channels} sample_width={self.sample_width}B "
            f"sample_rate={self.sample_rate}Hz "
            f"frames(declared={self.declared_frames}, actual={self.actual_frames})"
            f"{mismatch} duration={self.duration_seconds:.2f}s "
            f"size={self.file_size_bytes}B{byte_order}{note}"
        )


def _iter_chunks_lenient(
    data: bytes, start: int, *, big_endian: bool
) -> Iterator[tuple[str, int, bytes]]:
    """Like ``_iter_chunks``, but for diagnostics only: yields ``(chunk_id,
    declared_size, body)`` where ``body`` is clamped to whatever bytes are
    actually present, instead of raising when a chunk's declared size runs
    past the end of the file. A truncated final chunk is yielded once (with
    a short ``body``) and then iteration stops, since a truncated write
    leaves no reliable position for a next chunk header."""
    pos = start
    end = len(data)
    while pos + 8 <= end:
        chunk_id = data[pos : pos + 4].decode("ascii", errors="replace")
        declared_size = int.from_bytes(data[pos + 4 : pos + 8], "big" if big_endian else "little")
        body_start = pos + 8
        body_end = body_start + declared_size
        body = data[body_start : min(body_end, end)]
        yield chunk_id, declared_size, body
        if body_end > end:
            return  # truncated — no reliable position for a next chunk
        pos = body_end + (declared_size & 1)


def _riff_wave_facts(raw: bytes, form_type: bytes, size: int) -> AudioFacts:
    if form_type != b"WAVE":
        return AudioFacts(
            "RIFF", "unknown", 0, 0, 0, 0, 0, 0.0, size, f"unexpected form {form_type!r}"
        )

    channels = sample_width = sample_rate = 0
    compression = "unknown"
    declared_bytes = actual_bytes = 0
    for chunk_id, declared_size, body in _iter_chunks_lenient(raw, 12, big_endian=False):
        if chunk_id == "fmt " and len(body) >= 16:
            audio_format, channels, sample_rate, _byte_rate, _block_align, bits = struct.unpack(
                "<HHIIHH", body[0:16]
            )
            sample_width = bits // 8
            compression = "PCM" if audio_format == 1 else f"format={audio_format}"
        elif chunk_id == "data":
            declared_bytes = declared_size
            actual_bytes = len(body)

    note = "" if channels and sample_width else "missing or short 'fmt ' chunk"
    frame_size = channels * sample_width
    declared_frames = declared_bytes // frame_size if frame_size else 0
    actual_frames = actual_bytes // frame_size if frame_size else 0
    duration = actual_frames / sample_rate if sample_rate else 0.0
    return AudioFacts(
        "RIFF",
        compression,
        channels,
        sample_width,
        sample_rate,
        declared_frames,
        actual_frames,
        duration,
        size,
        note,
    )


def _aiff_facts(raw: bytes, form_type: bytes, size: int) -> AudioFacts:
    container = "AIFC" if form_type == b"AIFC" else "AIFF"
    channels = sample_width = sample_rate = declared_frames = 0
    compression = "NONE"
    pcm_data = b""
    note = ""
    for chunk_id, _declared_size, body in _iter_chunks_lenient(raw, 12, big_endian=True):
        if chunk_id == "COMM":
            if len(body) < 18:
                note = "COMM chunk shorter than the required 18 bytes"
                continue
            try:
                channels, declared_frames, sample_size = struct.unpack(">hlh", body[0:8])
                sample_rate = round(_read_ieee_extended(body[8:18]))
            except (struct.error, ValueError) as exc:
                note = f"malformed COMM chunk: {exc}"
                continue
            sample_width = sample_size // 8 if sample_size % 8 == 0 else 0
            if form_type == b"AIFC" and len(body) >= 22:
                compression = body[18:22].decode("ascii", errors="replace")
        elif chunk_id == "SSND":
            if len(body) < 8:
                note = note or "SSND chunk shorter than the required 8 bytes"
                continue
            offset = int.from_bytes(body[0:4], "big")
            pcm_data = body[8 + offset :]

    frame_size = channels * sample_width
    actual_frames = len(pcm_data) // frame_size if frame_size else 0
    duration = actual_frames / sample_rate if sample_rate else 0.0
    if not note and not channels:
        note = "missing or malformed COMM/SSND chunk"

    # Fix round 3: byte-order detection + amplitude diagnostics, purely for
    # display here (no logging — normalize_speech_audio_to_wav does its own
    # detection, on the bytes it actually converts, and logs there). Only
    # meaningful for 16/24/32-bit PCM with enough header info to decode.
    byte_order_used = "n/a"
    byte_order_label = "n/a"
    byte_order_big_score: float | None = None
    byte_order_little_score: float | None = None
    byte_order_inconclusive = False
    rms = peak = clipped_fraction = None
    if sample_width in (2, 3, 4) and channels and sample_rate and pcm_data:
        label_big_endian = compression not in ("sowt", "SOWT")
        decision = _decide_byte_order(
            pcm_data,
            sample_width=sample_width,
            channels=channels,
            sample_rate=sample_rate,
            label_big_endian=label_big_endian,
        )
        chosen = _decode_pcm_samples(pcm_data, sample_width, big_endian=decision.big_endian)
        rms, peak, clipped_fraction = _amplitude_stats(chosen, sample_width)
        byte_order_used = _order_name(decision.big_endian)
        byte_order_label = _order_name(decision.label_big_endian)
        byte_order_big_score = decision.big_endian_score
        byte_order_little_score = decision.little_endian_score
        byte_order_inconclusive = decision.inconclusive

    return AudioFacts(
        container,
        compression,
        channels,
        sample_width,
        sample_rate,
        declared_frames,
        actual_frames,
        duration,
        size,
        note,
        byte_order_used=byte_order_used,
        byte_order_label=byte_order_label,
        byte_order_big_score=byte_order_big_score,
        byte_order_little_score=byte_order_little_score,
        byte_order_inconclusive=byte_order_inconclusive,
        rms=rms,
        peak=peak,
        clipped_fraction=clipped_fraction,
    )


def extract_audio_facts(raw: bytes) -> AudioFacts:
    """Best-effort diagnostic facts about raw audio bytes, RIFF/WAVE or
    AIFF/AIFF-C — pure and total: given any ``bytes``, it always returns an
    ``AudioFacts`` (container ``"unknown"`` and an explanatory ``note`` for
    anything it cannot parse), never raises. This is deliberately more
    lenient than ``normalize_speech_audio_to_wav``: it exists to describe
    exactly what was written, truncation and all, not to gate on it.
    """
    size = len(raw)
    if size < 12:
        return AudioFacts("unknown", "unknown", 0, 0, 0, 0, 0, 0.0, size, f"only {size} bytes")

    riff_id, form_type = raw[0:4], raw[8:12]
    if riff_id == b"RIFF":
        return _riff_wave_facts(raw, form_type, size)
    if riff_id == b"FORM" and form_type in (b"AIFF", b"AIFC"):
        return _aiff_facts(raw, form_type, size)
    return AudioFacts(
        "unknown",
        "unknown",
        0,
        0,
        0,
        0,
        0,
        0.0,
        size,
        f"unrecognized header {riff_id!r}/{form_type!r}",
    )


def _ensure_facts_logging_visible(logger: logging.Logger | None = None) -> None:
    """Make sure ``log.info(...)`` calls in this module reach the console
    even when nothing else has configured logging. ``logger`` (default: this
    module's own) lets ``assistant.load_asr_model`` apply the same treatment
    to its dedicated ASR-facts logger (fix round 4).

    ``uv run demo --all`` (``shared/demo.py``) calls each project's
    ``demo()`` directly — it never calls ``logging.basicConfig``, and
    Python's own last-resort handler drops anything below ``WARNING`` — so
    without this, the audio-facts line below would silently vanish from
    the heavy.yml log even though nothing raised. Attaches a plain stdout
    handler to *this module's own logger only* (not the root logger), so it
    doesn't change logging behaviour for any other project's demo() sharing
    the same `uv run demo --all` process; ``propagate = False`` also stops
    the line from printing twice when a caller (e.g. ``p10-meeting-assistant
    --demo --verbose``) *has* configured the root logger. Idempotent: only
    adds the handler once, even across many ``generate()`` calls.
    """
    target = log if logger is None else logger
    if target.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False


# =============================================================================
# Wait for asynchronous writes to finish (fix round 2)
# =============================================================================
#
# heavy.yml run 35082867199 supplied the missing evidence: on macOS,
# ``standup.wav`` came back as ``container=AIFC compression=twos channels=1
# sample_width=2B sample_rate=22050Hz frames(declared=118, actual=118)
# duration=0.01s size=4332B`` — internally consistent (declared == actual
# frames), so the AIFF/AIFF-C conversion itself is exonerated; this is
# hypothesis (b). 4332 bytes matches Apple's AudioFile layout: a header
# padded out to a 4096-byte ``FLLR`` filler chunk plus 236 bytes of PCM (118
# frames at 16-bit mono, ~5ms). ``NSSpeechSynthesizer`` writes the file
# asynchronously and pyttsx3's ``runAndWait()`` returns before that finishes
# — ``generate()`` was reading (and converting) a file mid-write. The
# constants and function below make it wait for the write to actually be
# done, and refuse to hand off a file that stabilizes but is still
# implausibly short for what it was supposed to contain.

STABILIZE_POLL_SECONDS = 0.25  # how often to re-check the file's size
STABILIZE_REQUIRED_STABLE_READS = 3  # consecutive unchanged reads before "final"
STABILIZE_TIMEOUT_SECONDS = 60.0  # give up and treat the engine as broken
# A conservative floor on synthesized speech: even fast, dense speech is
# implausible under this many seconds per word. The real Windows/SAPI5
# baseline (see the fix-round-2 report) is 73 words in ~34s of audio, about
# 0.47s/word — comfortably clear of this floor; the macOS incident above was
# 118 frames (~0.01s) for the same 73-word script, comfortably under it.
MIN_SECONDS_PER_WORD = 0.1


# Fix round 5: heavy.yml run 35088397743 hit a race in round 2's wait. The
# macOS file sat at its header-only size (4332B, 118 frames, 0.005s) for
# longer than the 3 x 0.25s stability window before NSSpeechSynthesizer
# wrote its first real audio buffer, so the header was declared "stable" and
# then rejected by the plausibility floor — Whisper never ran. (In rounds 2
# and 3 the full 27.6s file happened to arrive inside the window.) A file
# below the floor is therefore never "finished": the floor is part of the
# wait, and only the timeout ends it.


@dataclass(frozen=True)
class WaitOutcome:
    """The result of ``_wait_for_audio``: the file's final bytes and facts,
    whether it is ready, and how long waiting took (for the facts log line)."""

    ready: bool  # size stable AND duration at/above the floor
    size_stable: bool  # at the last poll, size unchanged for the required reads
    raw: bytes  # the file's bytes as last read
    facts: AudioFacts  # extract_audio_facts(raw)
    elapsed_seconds: float
    polls: int  # number of size reads taken


def _wait_for_audio(
    path: Path,
    *,
    min_duration_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> WaitOutcome:
    """Poll ``path`` (every ``STABILIZE_POLL_SECONDS``) until it is ready:
    its size unchanged for ``STABILIZE_REQUIRED_STABLE_READS`` consecutive
    reads *and* its measured duration at least ``min_duration_seconds``. A
    size-stable file below that floor keeps being polled — the engine may
    simply not have written its first audio buffer yet. Gives up after
    ``STABILIZE_TIMEOUT_SECONDS`` and returns ``ready=False``; the caller
    raises. The file's bytes are re-read and re-parsed only when its size
    changes. On a platform where the file is already complete (Windows/
    Linux), this costs ``(STABILIZE_REQUIRED_STABLE_READS - 1) *
    STABILIZE_POLL_SECONDS``.

    ``sleep``/``monotonic`` are injectable (default: the real ``time``
    functions) so tests can drive this without a real clock or real sleeps.
    """
    start = monotonic()
    deadline = start + STABILIZE_TIMEOUT_SECONDS
    last_size = -1
    stable_reads = 0
    polls = 0
    raw = b""
    facts: AudioFacts | None = None
    while True:
        size = path.stat().st_size
        polls += 1
        if size == last_size:
            stable_reads += 1
        else:
            stable_reads = 1
            last_size = size
            facts = None  # size changed — parse again when next needed
        size_stable = stable_reads >= STABILIZE_REQUIRED_STABLE_READS
        timed_out = monotonic() >= deadline
        if size_stable or timed_out:
            if facts is None:
                raw = path.read_bytes()
                facts = extract_audio_facts(raw)
            if size_stable and facts.duration_seconds >= min_duration_seconds:
                return WaitOutcome(True, True, raw, facts, monotonic() - start, polls)
            if timed_out:
                return WaitOutcome(False, size_stable, raw, facts, monotonic() - start, polls)
        sleep(STABILIZE_POLL_SECONDS)


SCRIPTS: dict[str, str] = {
    "standup.wav": (
        "Good morning team. Today we discussed three topics. "
        "First, the mobile app release timeline. "
        "Second, the customer feedback from last week. "
        "Third, the hiring plan for the new engineer. "
        "We decided to postpone the release to next Friday. "
        "We also decided to hire two backend developers. "
        "For action items: Anna will fix the login bug by Wednesday. "
        "Mark will prepare the release notes by Thursday. "
        "Sarah will schedule the interviews for next week."
    ),
    "budget.wav": (
        "Welcome everyone to the budget review meeting. "
        "We talked about the marketing spend, the cloud costs, and the travel budget. "
        "The team decided to cut the travel budget by twenty percent. "
        "We agreed to move the servers to a cheaper cloud region. "
        "Action items are as follows. "
        "David will renegotiate the cloud contract by the end of the month. "
        "Elena will send the updated budget spreadsheet to finance tomorrow."
    ),
}


def _init_engine() -> Any:
    """``pyttsx3.init()``, with a message naming the fix when the driver is missing.

    Returns ``Any``: pyttsx3 ships no type stubs, so its engine object's
    methods (``setProperty``, ``save_to_file``, ``runAndWait``) are untyped
    regardless of this function's own signature.
    """
    import pyttsx3  # noqa: PLC0415 — lazy heavy/optional import, see module docstring

    try:
        return pyttsx3.init()
    except Exception as exc:  # driver missing/unusable — message names the fix
        if platform.system() == "Linux":
            hint = "install it with `sudo apt-get install espeak-ng` (or your distro's equivalent)"
        else:
            hint = (
                "install your OS text-to-speech engine (SAPI5 ships with Windows, "
                "NSSpeechSynthesizer with macOS)"
            )
        raise TTSEngineUnavailableError(
            f"No text-to-speech engine is available (pyttsx3.init() failed: {exc}); {hint}."
        ) from exc


def generate(
    data_dir: Path = DATA_DIR,
    force: bool = False,
    only: Iterable[str] | None = None,
    facts_out: dict[str, AudioFacts] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Path]:
    """Synthesize the requested scripts (default: every script in ``SCRIPTS``) into
    ``data_dir``; return name -> path.

    ``only`` restricts which scripts get synthesized. ``demo()`` passes
    ``{"standup.wav"}`` because it never needs ``budget.wav`` — see the bug
    below, which this sidesteps entirely for the one path CI actually runs.

    Skips a file that already exists unless ``force=True``, so repeated calls
    (e.g. two consecutive ``--demo`` runs) reuse the exact same audio bytes.
    That is what keeps the Whisper transcription that follows deterministic
    between runs, which ``output/transcript.txt`` being byte-identical
    run-to-run depends on. WAV files are never tracked by git (see
    ``.gitignore``), so the cache lives only on the local machine.

    Each requested script gets its own ``pyttsx3.init()`` engine instance,
    fully synthesized (``save_to_file`` then ``runAndWait()``) before the next
    one starts. An earlier version queued every pending script on one shared
    engine (``save_to_file`` called once per script, a single trailing
    ``runAndWait()``): on the eSpeak driver (Linux CI), that let the engine's
    ctypes callback proxy for one utterance get garbage-collected while a
    later utterance was still in flight (``ReferenceError: weakly-referenced
    object no longer exists``), and the corresponding WAV file was silently
    never written — no exception, just one file missing, discovered only when
    Whisper tried to open it and failed with "No such file or directory". One
    engine per file avoids the overlap that triggers this.

    As a last line of defence — in case this or some other driver/version
    combination still drops a file silently — every requested file is
    checked to exist and be non-empty after synthesis. If pyttsx3 reported no
    error but a file is still missing, this raises
    ``TTSEngineUnavailableError``, the same exception a genuinely absent
    engine raises, so callers already handling "no engine at all" (see
    ``demo()``) also handle "engine present but silently incomplete" the same
    clean way: a ``skipped`` result with an actionable note, never a raw
    ``ValueError`` two steps removed from the real cause.

    ``facts_out``, if given, is populated (in place) with an ``AudioFacts``
    per freshly-synthesized file (fix round 1) — the same facts this
    function logs at INFO, made available to callers (``demo()``) that need
    them in a failure note. Only covers files actually synthesized this
    call; a file reused from a previous run (see above) is not re-inspected.

    Fix round 2: ``runAndWait()`` returning is not proof the file is
    finished — macOS's driver writes asynchronously (see
    the module-level comments above ``STABILIZE_POLL_SECONDS`` and
    ``WaitOutcome`` for the heavy.yml evidence). Each freshly-synthesized
    file is polled (``_wait_for_audio``) until its size stops changing *and*
    its duration reaches the plausibility floor for its script's word count
    (``MIN_SECONDS_PER_WORD``, fix round 5) *before* any conversion happens;
    a file that is still changing or still below the floor when
    ``STABILIZE_TIMEOUT_SECONDS`` expires raises
    ``TTSEngineUnavailableError`` with the facts in the message — the
    engine is at fault, not this code, so ``demo()`` degrades to
    ``skipped``. ``sleep``/``monotonic`` are injectable so tests never
    really sleep.
    """
    names = set(SCRIPTS) if only is None else set(only)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: data_dir / name for name in names}
    pending = {name: SCRIPTS[name] for name in names if force or not paths[name].exists()}
    if not pending:
        return paths

    for name, text in pending.items():
        engine = _init_engine()
        engine.setProperty("rate", 165)
        engine.save_to_file(text, str(paths[name]))
        engine.runAndWait()

    missing = sorted(
        name for name in pending if not paths[name].exists() or paths[name].stat().st_size == 0
    )
    if missing:
        raise TTSEngineUnavailableError(
            "pyttsx3 reported no error but did not write "
            f"{', '.join(missing)} (seen on some eSpeak driver versions when synthesizing "
            "multiple utterances in one process); treating the text-to-speech engine as "
            "unavailable rather than returning a path with no audio behind it."
        )

    _ensure_facts_logging_visible()
    for name in pending:
        path = paths[name]

        # Fix round 2: wait for the OS engine to actually finish writing —
        # runAndWait() returning is not proof of that on macOS. Only once
        # the file's size has stopped changing is it safe to read facts
        # from it or convert it; reading mid-write is exactly what produced
        # heavy.yml run 35082867199's 118-frame/5ms file.
        # Fix round 5: the plausibility floor is part of the wait — a
        # size-stable file that is still below it keeps being polled (see
        # WaitOutcome's module comment).
        word_count = len(SCRIPTS[name].split())
        min_duration = MIN_SECONDS_PER_WORD * word_count
        outcome = _wait_for_audio(
            path, min_duration_seconds=min_duration, sleep=sleep, monotonic=monotonic
        )

        # Fix round 1: log one line of diagnostic facts per freshly-
        # synthesized file, read from its bytes exactly as the TTS engine
        # wrote them (as last read by the wait) — RIFF included, as a
        # Linux/Windows baseline, so a macOS run's numbers have something to
        # be compared against. Deliberately read *before* normalization
        # below, since that step may trim a short SSND/data chunk to fit
        # rather than raise on it — the mismatch this is meant to catch.
        # Fix round 5: plus how long the wait took and how many polls.
        raw = outcome.raw
        facts = outcome.facts
        wait_line = f"wait={outcome.elapsed_seconds:.2f}s polls={outcome.polls}"
        log.info("%s %s", facts.format(name), wait_line)
        if facts_out is not None:
            facts_out[name] = facts

        if not outcome.ready:
            if outcome.size_stable:
                raise TTSEngineUnavailableError(
                    f"'{name}' timed out after {outcome.elapsed_seconds:.1f}s waiting for audio: "
                    f"its size is stable but it stayed below the plausibility floor for its "
                    f"{word_count}-word script ({facts.duration_seconds:.3f}s < "
                    f"{min_duration:.3f}s minimum, {MIN_SECONDS_PER_WORD}s/word) for the whole "
                    f"{STABILIZE_TIMEOUT_SECONDS:.0f}s timeout; {wait_line}; {facts.format(name)}"
                )
            raise TTSEngineUnavailableError(
                f"'{name}' never finished writing: timed out after "
                f"{outcome.elapsed_seconds:.1f}s (its size kept changing — the OS speech engine "
                "likely writes asynchronously and returned from runAndWait() too early); "
                f"{wait_line}; last read: {facts.format(name)}"
            )

        # macOS's driver (NSSpeechSynthesizer) writes AIFF/AIFF-C bytes to
        # the ``.wav`` path regardless of extension; normalize in place to a
        # standard RIFF WAV so downstream readers (scipy.io.wavfile) never
        # see a mislabelled file. A no-op on Windows/Linux, where the bytes
        # are already RIFF. Raises TTSEngineUnavailableError (same contract
        # as above) if the file is neither a WAV nor a convertible
        # AIFF/AIFF-C.
        normalized = normalize_speech_audio_to_wav(raw, source_name=name)
        if normalized != raw:
            path.write_bytes(normalized)

    return paths


def main() -> None:
    """Force-regenerate every script into ``./data`` and print the paths written."""
    for path in generate(force=True).values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
