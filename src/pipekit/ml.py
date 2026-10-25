"""
Model training on top of jobs: a training run is a Job whose output is a folder of checkpoints, and picking a model
is Select over those checkpoints. Nothing here is specific to one kind of learning (RL, supervised, fine-tuning).

  TrainRun               a training job: a command writing a run folder -> a "run" artifact listing its checkpoints
  checkpoints_in         a run folder's checkpoints in step order (unnumbered ones - best, final - last)
  checkpoint_candidates  the same, as Select's candidates (pick the best checkpoint of a run)
"""

from __future__ import annotations

import os
import re
from typing import ClassVar

from pipekit.core import Artifact, Context
from pipekit.jobs import Job
from pipekit.steps import Candidates

# the training step in a checkpoint's file name (model_1000000_steps.zip, model_0000004600.pt): its last number
LAST_NUMBER = r"(\d+)\D*$"
CHECKPOINT_SUFFIXES = (".zip", ".pt", ".pth", ".ckpt", ".safetensors")


def checkpoints_in(folder: str, suffixes: tuple[str, ...] = CHECKPOINT_SUFFIXES, step_pattern: str = LAST_NUMBER,
                   root: str | None = None) -> list[str]:
    """The checkpoints in `folder` (files ending in one of `suffixes`) as <folder>/<file>, in training-step order:
    files `step_pattern` finds a number in first, by that number, then the rest (best, final, ...) by name. A relative
    `folder` is looked up under `root` (default: the working directory) but returned as given."""
    full = folder if root is None or os.path.isabs(folder) else os.path.join(root, folder)
    if not os.path.isdir(full):
        return []
    pattern = re.compile(step_pattern)

    def order(name: str) -> tuple[int, int, str]:
        found = pattern.search(name)
        return (0, int(found.group(1)), name) if found else (1, 0, name)

    files = sorted((f for f in os.listdir(full) if f.endswith(suffixes)), key=order)
    return [os.path.join(folder, f) for f in files]


def checkpoint_candidates(suffixes: tuple[str, ...] = CHECKPOINT_SUFFIXES, step_pattern: str = LAST_NUMBER,
                          include: str | None = None) -> Candidates:
    """Select candidates: every checkpoint of a run folder (only those whose file name contains `include`, if given);
    a source that is not a folder is its own single candidate."""

    def candidates(ctx: Context, source: str) -> list[str]:
        if not os.path.isdir(ctx.path(source)):
            return [source]
        found = checkpoints_in(source, suffixes, step_pattern, root=ctx.root)
        return [c for c in found if include is None or include in os.path.basename(c)]

    return candidates


class TrainRun(Job):  # pylint: disable=abstract-method
    """A training job: a Job whose output is a run folder (`run_dir`, the same as `out`), logged to train.log inside
    it, handed on as a "run" artifact with its checkpoints listed in meta (output name <name>.run by default)."""

    kind: ClassVar[str] = "run"
    folder: ClassVar[bool] = True

    def __init__(self, name: str, run_dir: str, output: str | None = None) -> None:
        super().__init__(name, run_dir, output or f"{name}.run")

    @property
    def run_dir(self) -> str:
        """The run folder (another name for `out`)."""
        return self.out

    def describe(self) -> str:
        return f"{type(self).__name__} -> {self.run_dir}"

    def log_path(self, ctx: Context) -> str:
        return os.path.join(ctx.path(self.run_dir), "train.log")

    def artifact(self, ctx: Context) -> Artifact:
        return Artifact(self.kind, self.run_dir, meta={"checkpoints": checkpoints_in(self.run_dir, root=ctx.root)})
