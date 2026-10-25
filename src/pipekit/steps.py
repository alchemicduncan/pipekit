"""
Generic steps: Select keeps the best of several candidates by an Evaluator; Require stops the pipeline unless an
artifact's metric clears a bar.
"""

from __future__ import annotations

from collections.abc import Callable

from pipekit.core import Artifact, Context, Step
from pipekit.evaluate import Evaluator
from pipekit.refs import refs, resolve

# candidates from the resolved source reference (e.g. every checkpoint in a run's folder)
Candidates = Callable[[Context, str], list[str]]


def _single(_ctx: Context, source: str) -> list[str]:
    return [source]


class Select(Step):
    """Score candidates and keep the best (Strategy: the evaluator) -> artifact `name` of `kind`.

    The candidates come from `source` ("@name" or a path) through `candidates` (default: just that path), plus
    `extra` ones (e.g. the starting point, which the new candidates must beat)."""

    def __init__(self, name: str, source: str, evaluator: Evaluator, candidates: Candidates = _single,
                 extra: tuple[str, ...] = (), kind: str = "checkpoint") -> None:
        super().__init__(name)
        self.source, self.evaluator, self.candidates, self.extra, self.kind = source, evaluator, candidates, extra, kind
        self.inputs = refs(source, *extra)
        self.outputs = (name,)

    def describe(self) -> str:
        return f"pick the best of {self.source} by {self.evaluator.name}"

    def execute(self, ctx: Context) -> dict[str, Artifact]:
        source = str(resolve(ctx, self.source))
        pool = self.candidates(ctx, source) + [str(resolve(ctx, e)) for e in self.extra]
        if ctx.dry_run:
            return {self.name: Artifact(self.kind, pool[-1] if pool else source, meta={"dry_run": True})}
        if not pool:
            raise RuntimeError(f"{self.name}: no candidates from {self.source}")
        report = self.evaluator.evaluate(pool)
        best = max(pool, key=lambda c: self.evaluator.score(report[c]))
        metrics = {**report[best], "score": self.evaluator.score(report[best])}
        return {self.name: Artifact(self.kind, best, metrics=metrics,
                                    meta={"evaluator": self.evaluator.name, "all": report})}


class Require(Step):
    """Stop the pipeline unless artifact.metrics[metric] >= minimum and/or <= maximum (a gate)."""

    def __init__(self, name: str, artifact: str, metric: str, minimum: float | None = None,
                 maximum: float | None = None) -> None:
        super().__init__(name)
        self.artifact, self.metric, self.minimum, self.maximum = artifact, metric, minimum, maximum
        self.inputs = (artifact,)

    def describe(self) -> str:
        bars = ([f">= {self.minimum:g}"] if self.minimum is not None else []) + \
               ([f"<= {self.maximum:g}"] if self.maximum is not None else [])
        return f"{self.artifact}.{self.metric} {' and '.join(bars)}"

    def execute(self, ctx: Context) -> dict[str, Artifact]:
        if not ctx.dry_run:
            value = ctx[self.artifact].metrics.get(self.metric)
            if value is None:
                raise RuntimeError(f"{self.artifact} has no metric {self.metric!r}: stopping here")
            if self.minimum is not None and value < self.minimum:
                raise RuntimeError(f"{self.artifact}.{self.metric} = {value:.3g} < {self.minimum:g}: stopping here")
            if self.maximum is not None and value > self.maximum:
                raise RuntimeError(f"{self.artifact}.{self.metric} = {value:.3g} > {self.maximum:g}: stopping here")
        return {}
