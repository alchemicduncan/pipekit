"""
External work as resumable steps: a Job runs a command that writes one output (a file or a folder) - a data export, a
preprocessing script, a build, a training run - and the step makes sure it is not done twice or adopted when stale.

  Job         launch a command that writes `out`; adopt an earlier attempt that finished; move a stale one aside when
              the step is forced (rerun, or its inputs were remade) -> an artifact for `out`
  set_aside   move a forced step's earlier output out of the way (kept, renamed with the time)
"""

from __future__ import annotations

import os
import time
from typing import ClassVar

from pipekit.core import Artifact, Context, Step
from pipekit.process import run_logged


def set_aside(ctx: Context, step: Step, path: str) -> str | None:
    """A forced step (rerun, or its inputs remade) must not adopt an earlier attempt's output: move `path` aside (kept,
    renamed <path>.old-<time>) so the step starts clean. Returns where it went, or None if nothing moved."""
    full = ctx.path(path)
    if step.name not in ctx.forced or ctx.dry_run or not os.path.exists(full):
        return None
    moved = f"{full}.old-{time.strftime('%Y%m%d-%H%M%S')}"
    os.replace(full, moved)
    return moved


class Job(Step):
    """A step whose work is an external command writing `out` (Template Method): execute() moves a stale
    attempt aside if forced, launches the command unless an earlier attempt finished, marks it finished, and returns
    artifact(). Subclasses give command() (or override launch() for work of several commands) and, if they like, where
    it logs, how it is judged finished and what the artifact holds.

    `out` is relative to the context's root. With `folder` True it is a folder: the done marker and the log go inside
    it. Otherwise it is a file: they go beside it (<out>.done, <out>.log). With done_marker None, override
    finished()."""

    kind: ClassVar[str] = "file"
    folder: ClassVar[bool] = False
    done_marker: ClassVar[str | None] = ".done"

    def __init__(self, name: str, out: str, output: str | None = None) -> None:
        super().__init__(name)
        self.out = out
        self.outputs = (output or name,)

    def describe(self) -> str:
        return f"{type(self).__name__} -> {self.out}"

    def command(self, ctx: Context) -> list[str]:
        """The command line."""
        raise NotImplementedError(f"{type(self).__name__} gives neither command() nor launch()")

    def _beside(self, ctx: Context, name: str) -> str:
        """A file that belongs to the output: inside it (a folder) or next to it (a file)."""
        full = ctx.path(self.out)
        return os.path.join(full, name) if self.folder else f"{full}{name}"

    def log_path(self, ctx: Context) -> str:
        """Where the command's output goes (appended)."""
        return self._beside(ctx, "job.log" if self.folder else ".log")

    def cwd(self, ctx: Context) -> str | None:
        """The working directory (default: the context's root)."""
        return ctx.root

    def env(self, ctx: Context) -> dict[str, str] | None:  # pylint: disable=unused-argument
        """Environment variables added for the command."""
        return None

    def finished(self, ctx: Context) -> bool:
        """Whether an earlier attempt completed (its output is adopted rather than made again)."""
        if self.done_marker is None:
            raise NotImplementedError(f"{type(self).__name__} has no done_marker: override finished()")
        return os.path.exists(self._beside(ctx, self.done_marker))

    def launch(self, ctx: Context) -> None:
        """Run the work to completion (raises if it fails)."""
        run_logged(self.command(ctx), self.log_path(ctx), env=self.env(ctx), cwd=self.cwd(ctx))

    def mark_finished(self, ctx: Context) -> None:
        """Record that the work completed."""
        if self.done_marker is None:
            return
        marker = self._beside(ctx, self.done_marker)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w", encoding="utf-8"):
            pass

    def artifact(self, ctx: Context) -> Artifact:  # pylint: disable=unused-argument
        """What the step hands on (default: `out`, as an artifact of `kind`)."""
        return Artifact(self.kind, self.out)

    def execute(self, ctx: Context) -> dict[str, Artifact]:
        set_aside(ctx, self, self.out)
        if self.done_marker is not None and not self.folder:
            set_aside(ctx, self, self.out + self.done_marker)  # a file's marker sits beside it: it goes too
        if not ctx.dry_run and not self.finished(ctx):
            self.launch(ctx)
            self.mark_finished(ctx)
        return {self.outputs[0]: self.artifact(ctx)}
