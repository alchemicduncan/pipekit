"""Shared test steps: ones that write a file, and a listener that records events."""

from __future__ import annotations

import os

import pytest

from pipekit.core import Artifact, Context, Listener, Step
from pipekit.refs import refs, resolve


class Touch(Step):
    """Writes <name>.txt (with its source's path inside, if any) and counts its executions."""

    def __init__(self, name: str, source: str | None = None, kind: str = "file") -> None:
        super().__init__(name)
        self.source, self.kind, self.calls = source, kind, 0
        self.inputs = refs(source) if source else ()
        self.outputs = (name,)

    def execute(self, ctx: Context) -> dict[str, Artifact]:
        self.calls += 1
        path = f"{self.name}.txt"
        if not ctx.dry_run:
            with open(ctx.path(path), "w", encoding="utf-8") as f:
                f.write(str(resolve(ctx, self.source)) if self.source else "")
        return {self.name: Artifact(self.kind, path, metrics={"calls": self.calls})}


class Boom(Step):
    """Always fails."""

    def execute(self, ctx: Context) -> dict[str, Artifact]:
        raise ValueError("boom")


class Recorder(Listener):
    """Keeps (event, step name) pairs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def step_started(self, step: Step) -> None:
        self.events.append(("started", step.name))

    def step_finished(self, step: Step, outputs: dict[str, Artifact], seconds: float) -> None:
        self.events.append(("finished", step.name))

    def step_skipped(self, step: Step, reason: str) -> None:
        self.events.append(("skipped", step.name))

    def step_failed(self, step: Step, error: BaseException) -> None:
        self.events.append(("failed", step.name))


@pytest.fixture
def root(tmp_path: object) -> str:
    """A scratch root with a state/ folder."""
    path = str(tmp_path)
    os.makedirs(os.path.join(path, "state"), exist_ok=True)
    return path
