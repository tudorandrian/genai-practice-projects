"""synthetic_audio.py — synthesize short meeting recordings for testing (offline).

Project P10. Uses pyttsx3 (Windows SAPI5 / eSpeak on Linux / NSSpeechSynthesizer
on macOS) to turn two short meeting scripts — each with clear topics, decisions
and action items — into WAV files under ``./data``. This keeps the project
self-contained: no copyrighted or expiring demo downloads, and the transcript
content is known ahead of time so the Whisper -> LLM chain can be checked end
to end.

``generate()`` does three things after pyttsx3 returns, because the OS engines
differ:

1. It waits until the file is plausibly long for its script and its size has
   stopped changing, since macOS writes asynchronously and ``runAndWait()``
   returns before the audio is on disk.
2. It logs one line of facts about the file exactly as written (container,
   format, frame counts, duration, and for AIFF input the byte-order and
   amplitude diagnostics), so a bad file on a machine nobody can inspect
   still explains itself in the log.
3. It converts AIFF/AIFF-C to a standard RIFF WAV in place, since macOS writes
   AIFF-C whatever the extension and ``assistant.load_audio`` reads RIFF only.

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
    """Raised when the OS text-to-speech engine cannot produce usable audio.

    ``pyttsx3`` wraps a per-OS driver (SAPI5 on Windows, ``espeak``/
    ``espeak-ng`` on Linux, ``NSSpeechSynthesizer`` on macOS) rather than
    shipping its own. It is raised when that driver is missing, when it
    reports success without writing a usable file, and when the file it wrote
    cannot be converted to WAV. Callers — notably ``demo()`` — catch this to
    degrade to a ``skipped`` result with an actionable message instead of a
    raw traceback or a ``failed`` status.
    """


# =============================================================================
# AIFF/AIFF-C parsing
# =============================================================================
#
# Python 3.13 removed the stdlib ``aifc`` module (PEP 594), so AIFF is parsed
# here with ``struct``: chunk sizes are big-endian, odd chunk bodies are padded
# to an even length, and the sample rate in ``COMM`` is a big-endian 80-bit
# IEEE-754 extended-precision float.


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
# Byte-order check (diagnostic only)
# =============================================================================
#
# The AIFF-C compression tag says the byte order ('twos' = big-endian, 'sowt'
# = little-endian), and the conversion always follows it. The signal is also
# scored both ways, as a diagnostic: speech decoded in the right order is
# smooth, while the wrong order looks like noise. A conclusive disagreement is
# logged as a WARNING and nothing else; the header is not overridden, because
# a heuristic should not silently rewrite a documented format field.


def _decode_pcm_samples(pcm: bytes, sample_width: int, *, big_endian: bool) -> np.ndarray:
    """Decode raw signed PCM bytes as a 1-D array of sample values (one
    entry per sample per channel — channels are not de-interleaved, which
    the scoring below does not need). Supports the sample widths AIFF/AIFF-C
    can declare: 1 (8-bit, signed, no byte order), 2, 3 (24-bit, assembled by
    hand since neither numpy nor ``struct`` has an int24 type) and 4 bytes."""
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
    mean absolute sample value. Speech at 22 kHz decoded in the right byte
    order is smooth relative to its own amplitude; decoded in the wrong
    order it is close to white noise, so the two scores differ widely."""
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


# The noisier score must be at least this many times the smoother one before
# the check counts as conclusive; near-silence, very short clips and white
# noise stay "inconclusive" instead of producing a guess.
BYTE_ORDER_DETECTION_MARGIN = 1.5
# Only a bounded prefix is scored: a few seconds tells smooth speech from
# noise, and scoring a long recording in full is needless work.
BYTE_ORDER_DETECTION_MAX_SECONDS = 5.0


@dataclass(frozen=True)
class ByteOrderCheck:
    """Both smoothness scores for 16/24/32-bit PCM, and what they suggest
    about byte order, next to what the header's label says."""

    label_big_endian: bool  # what the header/compression tag says (always used)
    big_endian_score: float  # smoothness decoding as big-endian (lower = smoother)
    little_endian_score: float  # smoothness decoding as little-endian
    detected_big_endian: bool | None  # None: neither order convincingly smoother

    @property
    def disagrees_with_label(self) -> bool:
        return (
            self.detected_big_endian is not None
            and self.detected_big_endian != self.label_big_endian
        )


def _check_byte_order(
    pcm: bytes,
    *,
    sample_width: int,
    channels: int,
    sample_rate: int,
    label_big_endian: bool,
) -> ByteOrderCheck:
    """Score ``pcm`` (16/24/32-bit signed PCM) decoded both ways over a
    bounded prefix (``BYTE_ORDER_DETECTION_MAX_SECONDS``). The order whose
    decoding is at least ``BYTE_ORDER_DETECTION_MARGIN`` times smoother is
    reported as detected; otherwise detection is ``None``. Pure — no I/O,
    no logging — and it decides nothing: callers always use the label."""
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
    detected = better_big if convincing else None
    return ByteOrderCheck(label_big_endian, big_score, little_score, detected)


def _order_name(big_endian: bool) -> str:
    return "big" if big_endian else "little"


def normalize_speech_audio_to_wav(raw: bytes, *, source_name: str) -> bytes:
    """Pure conversion: AIFF/AIFF-C bytes -> standard RIFF PCM WAV bytes.

    Takes and returns raw bytes only (no file I/O), so it is testable with no
    TTS engine, no OS driver, and no fixture files.

    - ``RIFF....WAVE`` -> returned byte-identical (already a standard WAV).
    - ``FORM....AIFF`` (always big-endian PCM) or ``FORM....AIFC`` with an
      uncompressed type (``NONE``/``twos`` = big-endian, ``sowt`` =
      little-endian) -> a little-endian RIFF PCM WAV with the same channel
      count, sample width and sample rate. The byte order is the one the
      header states. For 16/24/32-bit samples the signal is also checked
      (``_check_byte_order``), and a conclusive disagreement with the header
      is logged as a ``WARNING``; the header's order is still used. AIFF's
      8-bit PCM is signed and WAV's is unsigned, so 8-bit samples are shifted
      by 128. A trailing partial frame is dropped.
    - Anything else (unrecognized header, a compressed AIFC type, or
      malformed/truncated chunks) raises ``TTSEngineUnavailableError`` naming
      ``source_name`` and what was found, rather than handing a mislabelled
      file downstream.
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
                if form_type == b"AIFC":
                    if len(body) < 22:
                        raise ValueError(
                            "AIFF-C COMM chunk is shorter than the required 22 bytes "
                            "(its compression type is missing)"
                        )
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
    frame_size = channels * sample_width
    pcm = ssnd_data[: frame_count * frame_size]
    # A truncated SSND can end mid-frame; WAV's header must describe whole frames.
    pcm = pcm[: (len(pcm) // frame_size) * frame_size] if frame_size > 0 else b""

    if sample_width == 1:
        # AIFF 8-bit PCM is signed; WAV 8-bit PCM is unsigned — shift by 128.
        pcm = bytes((byte + 128) & 0xFF for byte in pcm)
    else:
        check = _check_byte_order(
            pcm,
            sample_width=sample_width,
            channels=channels,
            sample_rate=sample_rate,
            label_big_endian=label_big_endian,
        )
        if check.disagrees_with_label:
            log.warning(
                "%s: the signal looks %s-endian but the header says %s-endian; using the "
                "header (smoothness scores: big=%.4f little=%.4f, lower is smoother)",
                source_name,
                _order_name(not label_big_endian),
                _order_name(label_big_endian),
                check.big_endian_score,
                check.little_endian_score,
            )
        if label_big_endian:
            pcm = _byteswap(pcm, sample_width)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


# =============================================================================
# Audio facts
# =============================================================================
#
# Read from the bytes exactly as the engine wrote them, before conversion
# trims anything, so a truncated or implausible file shows up in the log
# instead of being hidden by the conversion.


@dataclass(frozen=True)
class AudioFacts:
    """Best-effort diagnostic snapshot of one audio file's header, read from
    its raw bytes exactly as written to disk (before
    ``normalize_speech_audio_to_wav`` trims a short ``SSND``/``data`` chunk)."""

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
    # Byte-order check and amplitude diagnostics: populated only for
    # AIFF/AIFF-C sources with sample_width in (2, 3, 4). RIFF/WAV and 8-bit
    # sources have no byte order to check, so these keep their defaults.
    byte_order_label: str = "n/a"  # "big"/"little" from the header — the order used
    byte_order_detected: str = "n/a"  # "big", "little" or "inconclusive"
    byte_order_big_score: float | None = None  # lower = smoother
    byte_order_little_score: float | None = None
    rms: float | None = None  # of the header-order decoding, as a fraction of full scale
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
        if self.byte_order_label != "n/a":
            flag = (
                " SIGNAL-DISAGREES"
                if self.byte_order_detected in ("big", "little")
                and self.byte_order_detected != self.byte_order_label
                else ""
            )
            byte_order = (
                f" byte_order={self.byte_order_label} detected={self.byte_order_detected}{flag} "
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

    # Byte-order check and amplitude diagnostics, for display only (the
    # conversion runs its own check on the bytes it converts, and logs there).
    byte_order_label = "n/a"
    byte_order_detected = "n/a"
    byte_order_big_score: float | None = None
    byte_order_little_score: float | None = None
    rms = peak = clipped_fraction = None
    if sample_width in (2, 3, 4) and channels and sample_rate and pcm_data:
        label_big_endian = compression not in ("sowt", "SOWT")
        check = _check_byte_order(
            pcm_data,
            sample_width=sample_width,
            channels=channels,
            sample_rate=sample_rate,
            label_big_endian=label_big_endian,
        )
        used = _decode_pcm_samples(pcm_data, sample_width, big_endian=label_big_endian)
        rms, peak, clipped_fraction = _amplitude_stats(used, sample_width)
        byte_order_label = _order_name(label_big_endian)
        byte_order_detected = (
            "inconclusive"
            if check.detected_big_endian is None
            else _order_name(check.detected_big_endian)
        )
        byte_order_big_score = check.big_endian_score
        byte_order_little_score = check.little_endian_score

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
        byte_order_label=byte_order_label,
        byte_order_detected=byte_order_detected,
        byte_order_big_score=byte_order_big_score,
        byte_order_little_score=byte_order_little_score,
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
    """Make sure ``logger``'s INFO lines (default: this module's own logger)
    reach the console even when nothing else has configured logging.
    ``assistant.load_asr_model`` uses it for its Whisper facts logger too.

    ``uv run demo --all`` (``shared/demo.py``) calls each project's
    ``demo()`` directly and never calls ``logging.basicConfig``, and Python's
    last-resort handler drops anything below ``WARNING``, so without this
    the facts lines would not appear. Attaches a plain stdout handler to
    that one logger only (not the root logger), so other projects' logging
    in the same process is unchanged; ``propagate = False`` stops the line
    printing twice when a caller (e.g. ``--demo --verbose``) has configured
    the root logger. Idempotent: adds the handler once.
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
# Waiting for the engine to finish writing
# =============================================================================
#
# macOS's NSSpeechSynthesizer writes the file asynchronously, and pyttsx3's
# runAndWait() returns before it is done. The engine can also sit on a
# header-only file for longer than the stability window before its first
# audio buffer arrives, so a file below the plausibility floor is never
# treated as finished: only the timeout ends the wait. On platforms whose
# file is already complete, the wait costs two poll intervals.

STABILIZE_POLL_SECONDS = 0.25  # how often to re-check the file's size
STABILIZE_REQUIRED_STABLE_READS = 3  # consecutive unchanged reads before "final"
STABILIZE_TIMEOUT_SECONDS = 60.0  # give up and treat the engine as broken
# A conservative floor on synthesized speech: even fast, dense speech is
# implausible under this many seconds per word. SAPI5 on Windows speaks the
# 73-word standup script in about 34 s (about 0.47 s/word), well above it; a
# header-only file (a few milliseconds) is far below it.
MIN_SECONDS_PER_WORD = 0.1


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
    import pyttsx3  # lazy heavy/optional import, see module docstring

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
    ``{"standup.wav"}`` because it never needs ``budget.wav``, which also
    keeps the eSpeak multi-utterance problem described below off that path.

    Skips a file that already exists unless ``force=True``, so repeated calls
    (e.g. two consecutive ``--demo`` runs) reuse the exact same audio bytes.
    That is what keeps the Whisper transcription that follows deterministic
    between runs, which ``output/transcript.txt`` being byte-identical
    run-to-run depends on. WAV files are never tracked by git (see
    ``.gitignore``), so the cache lives only on the local machine.

    Each requested script gets its own ``pyttsx3.init()`` engine instance,
    fully synthesized (``save_to_file`` then ``runAndWait()``) before the next
    one starts. Queuing every script on one shared engine let the eSpeak
    driver garbage-collect one utterance's ctypes callback while another was
    still in flight (``ReferenceError: weakly-referenced object no longer
    exists``), and that file was silently never written.

    After synthesis, for each file:

    - It must exist and be non-empty; otherwise ``TTSEngineUnavailableError``
      (the same exception a missing engine raises, so ``demo()`` reports a
      clean ``skipped``).
    - It is polled (``_wait_for_audio``) until its size stops changing *and*
      its duration reaches ``MIN_SECONDS_PER_WORD`` times its script's word
      count, within ``STABILIZE_TIMEOUT_SECONDS``; on timeout,
      ``TTSEngineUnavailableError`` with the facts in the message.
    - One facts line (``AudioFacts.format`` plus the wait's elapsed time and
      poll count) is logged at INFO, and stored in ``facts_out`` if given.
      A file reused from a previous run is not re-inspected.
    - It is converted in place to a RIFF WAV if the engine wrote AIFF/AIFF-C.

    If any of those steps raises, every file this call synthesized and had
    not yet finished is deleted before the exception propagates. Otherwise
    the next call would find a bad file, skip synthesis, and fail later in
    ``load_audio`` instead of reporting ``skipped``.

    ``sleep``/``monotonic`` are injectable so tests never really sleep.
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

    finished: set[str] = set()
    try:
        _finish_synthesized_files(pending, paths, finished, facts_out, sleep, monotonic)
    except TTSEngineUnavailableError:
        for name in pending:
            if name not in finished:
                paths[name].unlink(missing_ok=True)
        raise
    return paths


def _finish_synthesized_files(
    pending: dict[str, str],
    paths: dict[str, Path],
    finished: set[str],
    facts_out: dict[str, AudioFacts] | None,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    """``generate()``'s post-synthesis steps: existence check, wait, facts
    line, conversion. Adds each name to ``finished`` once its file is final."""
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

        word_count = len(SCRIPTS[name].split())
        min_duration = MIN_SECONDS_PER_WORD * word_count
        outcome = _wait_for_audio(
            path, min_duration_seconds=min_duration, sleep=sleep, monotonic=monotonic
        )

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

        # macOS writes AIFF/AIFF-C to the .wav path; downstream reads RIFF only.
        # A no-op for RIFF input (Windows/Linux).
        normalized = normalize_speech_audio_to_wav(outcome.raw, source_name=name)
        if normalized != outcome.raw:
            path.write_bytes(normalized)
        finished.add(name)


def main() -> None:
    """Force-regenerate every script into ``./data`` and print the paths written."""
    for path in generate(force=True).values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
