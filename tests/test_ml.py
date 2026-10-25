"""ml: training runs and picking a run's best checkpoint."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import os
import re
import sys

from pipekit.core import Context, run_pipeline
from pipekit.evaluate import Evaluator, Metrics
from pipekit.ml import TrainRun, checkpoint_candidates, checkpoints_in
from pipekit.steps import Select


def touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


class LastNumber(Evaluator):
    """Scores a checkpoint by the last number in its name."""

    def evaluate(self, candidates: list[str]) -> dict[str, Metrics]:
        return {c: {"n": float(re.findall(r"\d+", c)[-1])} for c in candidates}

    def score(self, metrics: Metrics) -> float:
        return metrics["n"]


def test_checkpoints_in_step_order(tmp_path):
    run = tmp_path / "run"
    for f in ("best.zip", "m_200_steps.zip", "m_30_steps.zip", "model_0000004600.pt", "notes.txt", "final.zip"):
        touch(str(run / f))
    assert checkpoints_in(str(run)) == [str(run / f) for f in ("m_30_steps.zip", "m_200_steps.zip",
                                                               "model_0000004600.pt", "best.zip", "final.zip")]
    assert checkpoints_in("run", (".zip",), r"_(\d+)_steps", root=str(tmp_path)) == [
        "run/m_30_steps.zip", "run/m_200_steps.zip", "run/best.zip", "run/final.zip"]
    assert not checkpoints_in(str(tmp_path / "nowhere"))


def test_select_best_checkpoint_of_a_run(tmp_path):
    for f in ("c2.zip", "c9.zip", "c5.zip", "other_c7.zip"):
        touch(str(tmp_path / "run" / f))
    ctx = Context(str(tmp_path))
    step = Select("best", "run", LastNumber(), checkpoint_candidates((".zip",), include="c", step_pattern=r"c(\d+)"))
    assert step.execute(ctx)["best"].path == "run/c9.zip"
    assert checkpoint_candidates()(ctx, "run/c2.zip") == ["run/c2.zip"]
    assert checkpoint_candidates(include="other")(ctx, "run") == ["run/other_c7.zip"]


class Script(TrainRun):
    """Runs a python one-liner that writes a checkpoint into the run folder."""

    def __init__(self, name: str, run_dir: str, code: str = "") -> None:
        super().__init__(name, run_dir)
        self.code = code or f"open('{run_dir}/model_1.pt', 'w').close()"

    def command(self, ctx: Context) -> list[str]:
        return [sys.executable, "-c", self.code]


def test_train_run_launches_and_adopts(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "r"))
    ctx = run_pipeline(Script("t", "r"), root, listeners=[])
    art = ctx["t.run"]
    assert art.kind == "run" and art.meta["checkpoints"] == ["r/model_1.pt"]
    assert sorted(os.listdir(os.path.join(root, "r"))) == [".done", "model_1.pt", "train.log"]
    again = Script("t", "r", code="raise SystemExit(1)")  # finished earlier: adopted, not launched
    assert run_pipeline(again, root, listeners=[])["t.run"].path == "r"
    assert again.run_dir == again.out == "r" and again.describe() == "Script -> r"


def test_train_run_forced_starts_clean(tmp_path):
    root = str(tmp_path)
    touch(os.path.join(root, "r", ".done"))
    touch(os.path.join(root, "r", "stale.pt"))
    ctx = Context(root)
    ctx.forced.add("t")
    art = Script("t", "r").execute(ctx)["t.run"]  # the finished attempt is moved aside and the job runs again
    assert art.meta["checkpoints"] == ["r/model_1.pt"]
    assert any(name.startswith("r.old-") for name in os.listdir(root))


def test_train_run_dry_run(tmp_path):
    assert run_pipeline(Script("t", "r"), str(tmp_path), dry_run=True, listeners=[])["t.run"].meta == {
        "checkpoints": []}
