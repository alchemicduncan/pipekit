"""jobs: external commands as resumable steps, in a pipeline with no model in it."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import os
import sys

import pytest

from pipekit.core import Artifact, Context, Sequence, run_pipeline
from pipekit.jobs import Job, set_aside
from pipekit.refs import refs, resolve
from pipekit.steps import Require
from tests.conftest import Touch


def touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


class Python(Job):
    """Runs a python one-liner (with OUT set to its output's path)."""

    def __init__(self, name: str, out: str, code: str, source: str | None = None) -> None:
        super().__init__(name, out)
        self.code, self.source = code, source
        self.inputs = refs(source) if source else ()

    def command(self, ctx: Context) -> list[str]:
        return [sys.executable, "-c", self.code]

    def env(self, ctx: Context) -> dict[str, str]:
        return {"OUT": ctx.path(self.out), "SRC": ctx.path(str(resolve(ctx, self.source))) if self.source else ""}


class Count(Python):
    """A report: the rows in its source, as a metric."""

    kind = "report"

    def __init__(self, name: str, out: str, source: str) -> None:
        super().__init__(name, out, "import os, json; rows = open(os.environ['SRC']).read().split(); "
                                    "json.dump({'rows': len(rows)}, open(os.environ['OUT'], 'w'))", source)

    def artifact(self, ctx: Context) -> Artifact:
        if ctx.dry_run:
            return Artifact(self.kind, self.out)
        with open(ctx.path(self.out), encoding="utf-8") as f:
            return Artifact(self.kind, self.out, metrics=json.load(f))


def etl(code: str = "import os; open(os.environ['OUT'], 'w').write('a b c')") -> Sequence:
    return Sequence("etl", [Python("clean", "data/clean.txt", code), Count("count", "data/count.json", "@clean"),
                            Require("enough", "count", "rows", minimum=3)])


def test_a_data_pipeline_runs_and_resumes(tmp_path):
    root, state = str(tmp_path), str(tmp_path / "state")
    os.makedirs(os.path.join(root, "data"))
    ctx = run_pipeline(etl(), root, state, listeners=[])
    assert ctx["count"].metrics == {"rows": 3} and ctx["clean"].kind == "file"
    assert os.path.exists(os.path.join(root, "data", "clean.txt.done"))
    with open(os.path.join(root, "data", "clean.txt.log"), encoding="utf-8") as f:
        assert f.read().startswith("\n$ ")
    run_pipeline(etl(code="raise SystemExit(1)"), root, state, listeners=[])  # everything restored: nothing runs


def test_a_job_whose_marker_exists_is_adopted(tmp_path):
    root = str(tmp_path)
    touch(os.path.join(root, "x.txt"))
    touch(os.path.join(root, "x.txt.done"))
    assert Python("j", "x.txt", "raise SystemExit(1)").execute(Context(root))["j"].path == "x.txt"


def test_rerun_moves_the_file_and_its_marker_aside(tmp_path):
    root, state = str(tmp_path), str(tmp_path / "state")
    os.makedirs(os.path.join(root, "data"))
    run_pipeline(etl(), root, state, listeners=[])
    ctx = run_pipeline(etl(code="import os; open(os.environ['OUT'], 'w').write('a b c d')"), root, state,
                       listeners=[], rerun=("clean",))
    assert ctx["count"].metrics == {"rows": 4}  # remade, and the report after it (its input is new)
    old = sorted(f.split(".old-")[0] for f in os.listdir(os.path.join(root, "data")) if ".old-" in f)
    assert old == ["clean.txt", "clean.txt.done", "count.json", "count.json.done"]


def test_folder_job_keeps_its_marker_and_log_inside(tmp_path):
    class Build(Python):
        """Writes a folder."""

        folder = True

    root = str(tmp_path)
    code = "import os; os.makedirs(os.environ['OUT'], exist_ok=True); open(os.environ['OUT'] + '/page.html', 'w')"
    art = Build("site", "site", code).execute(Context(root))["site"]
    assert art.path == "site" and sorted(os.listdir(os.path.join(root, "site"))) == [".done", "job.log", "page.html"]


def test_failure_leaves_no_marker(tmp_path):
    root = str(tmp_path)
    with pytest.raises(RuntimeError, match="exit 2"):
        Python("j", "x.txt", "raise SystemExit(2)").execute(Context(root))
    assert not os.path.exists(os.path.join(root, "x.txt.done"))


def test_dry_run_and_defaults(tmp_path):
    root = str(tmp_path)
    job = Python("j", "x.txt", "raise SystemExit(1)")
    assert job.execute(Context(root, dry_run=True))["j"].path == "x.txt" and job.describe() == "Python -> x.txt"

    class NoCommand(Job):  # pylint: disable=abstract-method
        """Gives nothing."""

    class NoMarker(Job):  # pylint: disable=abstract-method
        """No marker and no finished()."""

        done_marker = None

    with pytest.raises(NotImplementedError, match="neither"):
        NoCommand("n", "r").execute(Context(root))
    with pytest.raises(NotImplementedError, match="done_marker"):
        NoMarker("n", "r").execute(Context(root))
    NoMarker("n", "r").mark_finished(Context(root))  # nothing to write
    assert NoCommand("n", "r").env(Context(root)) is None and NoCommand("n", "r").cwd(Context(root)) == root


def test_set_aside_only_when_forced(tmp_path):
    root = str(tmp_path)
    touch(os.path.join(root, "out", "x"))
    ctx, step = Context(root), Touch("s")
    assert set_aside(ctx, step, "out") is None
    ctx.forced.add("s")
    moved = set_aside(ctx, step, "out")
    assert moved is not None and not os.path.exists(os.path.join(root, "out")) and os.path.isdir(moved)
