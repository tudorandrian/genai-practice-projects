"""Committed proofs under projects/*/output/ must be byte-identical on every OS and every
git line-ending setting, so every text write is required to pin `newline="\n"` explicitly."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

ROOT = Path(__file__).resolve().parents[2]


def _source_files() -> list[Path]:
    files: list[Path] = []
    for base in (ROOT / "projects", ROOT / "shared"):
        for path in base.rglob("*.py"):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            files.append(path)
    return sorted(files)


def _has_newline_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "newline" for kw in call.keywords)


def _mode_constant(call: ast.Call, positional_index: int) -> str | None:
    """Return the string value of the call's mode argument, or None if it isn't a
    string constant (positional at `positional_index`, or a `mode=` keyword)."""
    node: ast.expr | None = None
    if len(call.args) > positional_index:
        node = call.args[positional_index]
    else:
        for kw in call.keywords:
            if kw.arg == "mode":
                node = kw.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _violation(call: ast.Call, path: Path) -> str | None:
    """Return a "file:line: reason" string if `call` is a text write missing
    `newline=`, else None."""
    if isinstance(call.func, ast.Attribute) and call.func.attr == "write_text":
        if not _has_newline_kwarg(call):
            return f"{path.relative_to(ROOT)}:{call.lineno}: write_text(...) without newline="
        return None

    is_builtin_open = isinstance(call.func, ast.Name) and call.func.id == "open"
    is_path_open = isinstance(call.func, ast.Attribute) and call.func.attr == "open"
    if not (is_builtin_open or is_path_open):
        return None

    mode = _mode_constant(call, 1 if is_builtin_open else 0)
    if mode is None or "b" in mode or not ("w" in mode or "a" in mode):
        return None
    if _has_newline_kwarg(call):
        return None
    return f"{path.relative_to(ROOT)}:{call.lineno}: open(mode={mode!r}) without newline="


def test_every_text_write_pins_lf_newline() -> None:
    """A write_text/open call in write or append mode without an explicit `newline=`
    translates "\n" to os.linesep - CRLF on Windows - which would make every
    `uv run demo` dirty the LF proof files committed under projects/*/output/."""
    violations = [
        v
        for path in _source_files()
        for call in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(call, ast.Call)
        if (v := _violation(call, path)) is not None
    ]
    assert not violations, 'text writes missing newline="\\n":\n' + "\n".join(violations)


def test_write_text_with_explicit_newline_never_emits_crlf(tmp_path: Path) -> None:
    """Documents the mechanism the guard above enforces: `newline="\n"` disables the
    platform translation that `Path.write_text` would otherwise apply."""
    path = tmp_path / "proof.txt"
    path.write_text("a\nb\n", encoding="utf-8", newline="\n")
    assert b"\r\n" not in path.read_bytes()
