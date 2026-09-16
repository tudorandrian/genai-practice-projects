import hashlib
from pathlib import Path

import pytest

from shared import datasets

pytestmark = pytest.mark.core


def test_fetch_returns_cached_file_when_checksum_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"a,b\n1,2\n"
    digest = hashlib.sha256(payload).hexdigest()
    calls: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(payload)

    monkeypatch.setattr(datasets, "_download", fake_download)
    first = datasets.fetch("tiny", "https://example.invalid/tiny.csv", digest, cache_dir=tmp_path)
    second = datasets.fetch("tiny", "https://example.invalid/tiny.csv", digest, cache_dir=tmp_path)
    assert first == second == tmp_path / "tiny.csv"
    assert calls == ["https://example.invalid/tiny.csv"]  # downloaded once


def test_fetch_ignores_url_query_string_when_naming_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"a,b\n1,2\n"
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(datasets, "_download", lambda url, dest: dest.write_bytes(payload))
    result = datasets.fetch(
        "q", "https://example.invalid/data.csv?token=abc", digest, cache_dir=tmp_path
    )
    assert result == tmp_path / "q.csv"


def test_fetch_rejects_wrong_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "_download", lambda url, dest: dest.write_bytes(b"x"))
    with pytest.raises(datasets.ChecksumError):
        datasets.fetch("bad", "https://example.invalid/bad.csv", "0" * 64, cache_dir=tmp_path)
    assert not (tmp_path / "bad.csv").exists()  # a mismatched download never lands at dest
    assert not (tmp_path / "bad.csv.part").exists()


def test_fetch_leaves_no_file_when_the_download_fails_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_download(url: str, dest: Path) -> None:
        dest.write_bytes(b"a,b,1,")  # some bytes arrive, then the connection drops
        raise OSError("connection reset")

    monkeypatch.setattr(datasets, "_download", broken_download)
    with pytest.raises(OSError, match="connection reset"):
        datasets.fetch("cut", "https://example.invalid/cut.csv", "0" * 64, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cache_dir_comes_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GENAI_DATA_DIR", str(tmp_path / "custom"))
    assert datasets.default_cache_dir() == tmp_path / "custom"
