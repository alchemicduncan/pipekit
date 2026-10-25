"""
Scoring candidates (Strategy): an Evaluator measures each candidate and turns its metrics into one number to rank by.
ScoreBlend weighs several evaluators into one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

Metrics = dict[str, float]


class Evaluator(ABC):
    """Measures candidates (paths: checkpoints, exports, ...) and ranks them."""

    @property
    def name(self) -> str:
        """For logs and plans."""
        return type(self).__name__

    @abstractmethod
    def evaluate(self, candidates: list[str]) -> dict[str, Metrics]:
        """Metrics per candidate."""

    @abstractmethod
    def score(self, metrics: Metrics) -> float:
        """One number to rank candidates by (higher is better)."""


class ScoreBlend(Evaluator):
    """Several evaluators at once; score = sum of weight x each one's score. Each part's metrics are kept, prefixed
    with its index ("0:success", "0:score", "1:...")."""

    def __init__(self, parts: Iterable[tuple[float, Evaluator]]) -> None:
        self.parts = list(parts)

    @property
    def name(self) -> str:
        return " + ".join(f"{w:g} x {e.name}" for w, e in self.parts)

    def evaluate(self, candidates: list[str]) -> dict[str, Metrics]:
        report: dict[str, Metrics] = {c: {} for c in candidates}
        for i, (_, evaluator) in enumerate(self.parts):
            for c, metrics in evaluator.evaluate(candidates).items():
                report[c].update({f"{i}:{k}": v for k, v in metrics.items()})
                report[c][f"{i}:score"] = evaluator.score(metrics)
        return report

    def score(self, metrics: Metrics) -> float:
        return sum(w * metrics[f"{i}:score"] for i, (w, _) in enumerate(self.parts))
