"""The twelve demo entries. Each project appends its own line in its port PR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["core", "models", "rag"]


@dataclass(frozen=True)
class Entry:
    slug: str
    tier: Tier
    target: str  # "package.module:function" returning shared.demo.DemoResult


ENTRIES: list[Entry] = [
    Entry("p01-mini-etl", "core", "projects.p01_mini_etl.mini_etl:demo"),
    Entry("p02-question-bank", "core", "projects.p02_question_bank.qbank:demo"),
    Entry("p03-regression", "core", "projects.p03_regression.regression:demo"),
]
