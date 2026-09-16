"""Download-with-checksum into a local cache. The only cross-project runtime helper."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    """The downloaded file does not match the expected SHA-256."""


def default_cache_dir() -> Path:
    env = os.environ.get("GENAI_DATA_DIR")
    return Path(env) if env else Path.home() / ".cache" / "genai-practice-projects"


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, sha256: str, cache_dir: Path | None = None) -> Path:
    """Return the local path of `name`, downloading from `url` once and verifying its checksum."""
    base = cache_dir or default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    dest = base / (name + Path(urlsplit(url).path).suffix)
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    # Download next to the destination and move it into place only once the checksum
    # passes, so an interrupted or corrupt download never leaves a file under `dest`.
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s -> %s", url, dest)
    try:
        _download(url, part)
        actual = _sha256(part)
        if actual != sha256:
            raise ChecksumError(f"{name}: expected {sha256[:12]}…, got {actual[:12]}…")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest
