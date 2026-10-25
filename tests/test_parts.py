"""refs, builders, evaluate, steps, overrides, process."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import os
import sys
from typing import Any, ClassVar

import pytest

from pipekit import overrides
from pipekit.builders import Recipe, Stage
from pipekit.core import Artifact, Context, Step, run_pipeline
from pipekit.evaluate import Evaluator, Metrics, ScoreBlend
from pipekit.process import CommandFailed, run_logged
from pipekit.refs import is_ref, refs, resolve
from pipekit.steps import Require, Select
from tests.conftest import Touch


# ---- refs

def test_refs_and_resolve(root):
    ctx = Context(root)
    ctx.put("a", Artifact("file", "a.txt"))
    assert refs("@a", "plain", 3, None, "@b") == ("a", "b")
    assert resolve(ctx, "@a") == "a.txt" and resolve(ctx, "plain") == "plain" and resolve(ctx, 5) == 5
    assert is_ref("@a") and not is_ref("a") and not is_ref(1)


# ---- builders

class Pair(Stage):
    """Two Touch steps, the second reading the first."""

    def __init__(self, name: str, first: str) -> None:
        self.first = first
        super().__init__(name)

    def build(self) -> list[Step]:
        return [Touch(self.first), Touch(f"{self.first}2", f"@{self.first}")]


class Projects(Recipe):  # pylint: disable=abstract-method
    """This test module's own registry."""

    registry: ClassVar[dict[str, type[Recipe]]] = {}


class Demo(Projects, key="demo"):
    """A demo pipeline."""

    def stages(self, **options: Any) -> list[Step]:
        return [Pair("pair", options.get("first", "x"))]


def test_stage_and_recipe_build_their_trees(root):
    recipe = Demo("t1", first="y")
    assert recipe.name == "demo[t1]" and recipe.tag == "t1"
    ctx = run_pipeline(recipe, root, listeners=[])
    assert "y2" in ctx
    assert Projects.registry == {"demo": Demo} and "demo" not in Recipe.registry


def test_duplicate_recipe_keys_are_rejected():
    with pytest.raises(ValueError, match="taken"):
        class Again(Projects, key="demo"):  # pylint: disable=unused-variable,abstract-method
            """Takes a key already taken."""


def test_unimplemented_builders_raise():
    with pytest.raises(NotImplementedError):
        Stage("s")
    with pytest.raises(NotImplementedError):
        Recipe("t")


# ---- evaluate and Select

class ByNumber(Evaluator):
    """Scores 'cN' as N (its metric 'n'), optionally scaled."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def evaluate(self, candidates: list[str]) -> dict[str, Metrics]:
        return {c: {"n": float(c.rsplit("c", 1)[-1])} for c in candidates}

    def score(self, metrics: Metrics) -> float:
        return self.scale * metrics["n"]


def test_score_blend():
    blend = ScoreBlend([(1.0, ByNumber()), (-2.0, ByNumber(0.5))])
    report = blend.evaluate(["c1", "c3"])
    assert report["c3"] == {"0:n": 3.0, "0:score": 3.0, "1:n": 3.0, "1:score": 1.5}
    assert blend.score(report["c3"]) == pytest.approx(0.0)
    assert blend.name == "1 x ByNumber + -2 x ByNumber"


def test_select_keeps_the_best_candidate(root):
    def listing(_ctx: Context, source: str) -> list[str]:
        return [f"{source}/c{i}" for i in (2, 7, 4)]

    ctx = Context(root)
    ctx.put("run", Artifact("folder", "runs/r"))
    ctx.put("start", Artifact("checkpoint", "start/c5"))
    step = Select("best", "@run", ByNumber(), listing, extra=("@start",))
    assert step.inputs == ("run", "start") and "ByNumber" in step.describe()
    best = step.execute(ctx)["best"]
    assert best.path == "runs/r/c7" and best.metrics["score"] == 7.0 and len(best.meta["all"]) == 4


def test_select_dry_run_and_empty(root):
    ctx = Context(root, dry_run=True)
    assert Select("best", "c1", ByNumber()).execute(ctx)["best"].meta == {"dry_run": True}
    with pytest.raises(RuntimeError, match="no candidates"):
        Select("best", "x", ByNumber(), lambda _c, _s: []).execute(Context(root))


# ---- Require

@pytest.mark.parametrize("value, minimum, maximum, ok", [
    (0.5, 0.4, None, True), (0.3, 0.4, None, False), (0.5, None, 0.6, True), (0.7, None, 0.6, False),
    (0.5, 0.4, 0.6, True),
])
def test_require(root, value, minimum, maximum, ok):
    ctx = Context(root)
    ctx.put("a", Artifact("report", "-", metrics={"m": value}))
    gate = Require("gate", "a", "m", minimum, maximum)
    if ok:
        assert not gate.execute(ctx)
    else:
        with pytest.raises(RuntimeError, match="stopping here"):
            gate.execute(ctx)


def test_require_missing_metric_and_dry_run(root):
    ctx = Context(root)
    ctx.put("a", Artifact("report", "-"))
    with pytest.raises(RuntimeError, match="no metric"):
        Require("g", "a", "m", 0.0).execute(ctx)
    assert not Require("g", "a", "m", 0.0).execute(Context(root, dry_run=True))
    assert Require("g", "a", "m", 0.1, 0.9).describe() == "a.m >= 0.1 and <= 0.9"


# ---- overrides

def test_overrides_round_trip_and_apply(root):
    sd = os.path.join(root, "st")
    assert overrides.load(sd) == {}
    overrides.save(sd, {"x": {"source": "@y", "kind": "other"}})
    tree = Pair("pair", "x")
    changes = overrides.apply(tree, overrides.load(sd))
    first = tree.steps[0]
    assert isinstance(first, Touch) and first.kind == "other" and len(changes) == 2


def test_overrides_merge_dicts_and_reject_typos():
    step = Touch("x")
    step.settings = {"a": 1, "b": 2}  # type: ignore[attr-defined]
    overrides.apply(step, {"x": {"settings": {"b": 3}}})
    assert step.settings == {"a": 1, "b": 3}  # type: ignore[attr-defined]
    with pytest.raises(KeyError, match="no step"):
        overrides.apply(step, {"nope": {}})
    with pytest.raises(AttributeError, match="no attribute"):
        overrides.apply(step, {"x": {"nope": 1}})


# ---- process

def test_run_logged(root):
    log = os.path.join(root, "logs", "out.log")
    run_logged([sys.executable, "-c", "import os; print('hi', os.environ['PK_TEST'])"], log, env={"PK_TEST": "1"})
    with pytest.raises(CommandFailed, match="exit 3") as err:
        run_logged([sys.executable, "-c", "raise SystemExit(3)"], log, cwd=root)
    assert err.value.returncode == 3
    with open(log, encoding="utf-8") as f:
        text = f.read()
    assert "hi 1" in text and text.count("\n$ ") == 2
