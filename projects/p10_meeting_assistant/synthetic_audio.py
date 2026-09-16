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

import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


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


def generate(data_dir: Path = DATA_DIR, force: bool = False) -> dict[str, Path]:
    """Synthesize every script in ``SCRIPTS`` into ``data_dir``; return name -> path.

    Skips a file that already exists unless ``force=True``, so repeated calls
    (e.g. two consecutive ``--demo`` runs) reuse the exact same audio bytes.
    That is what keeps the Whisper transcription that follows deterministic
    between runs, which ``output/transcript.txt`` being byte-identical
    run-to-run depends on. WAV files are never tracked by git (see
    ``.gitignore``), so the cache lives only on the local machine.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: data_dir / name for name in SCRIPTS}
    pending = {name: text for name, text in SCRIPTS.items() if force or not paths[name].exists()}
    if not pending:
        return paths

    import pyttsx3  # noqa: PLC0415 — lazy heavy/optional import, see module docstring

    try:
        engine = pyttsx3.init()
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
    engine.setProperty("rate", 165)
    for name, text in pending.items():
        engine.save_to_file(text, str(paths[name]))
    engine.runAndWait()
    return paths


def main() -> None:
    """Force-regenerate every script into ``./data`` and print the paths written."""
    for path in generate(force=True).values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
