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
    Entry("p04-decision-tree", "core", "projects.p04_decision_tree.decision_tree:demo"),
    Entry("p05-segmentation", "core", "projects.p05_segmentation.segmentation:demo"),
    Entry("p06-ml-pipeline", "core", "projects.p06_ml_pipeline.ml_pipeline:demo"),
    Entry("p07-sentiment-api", "core", "projects.p07_sentiment_api.server:demo"),
    Entry("p08-image-captioning", "models", "projects.p08_image_captioning.captioner:demo"),
    Entry("p09-chatbot", "models", "projects.p09_chatbot.engine:demo"),
]
