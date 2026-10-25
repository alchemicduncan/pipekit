"""core: steps, composites, the manifest, listeners, run_pipeline."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import os

import pytest

from pipekit.core import (Artifact, ConsoleListener, Context, JsonLogListener, Manifest, Parallel, Sequence,
                          format_metrics, leaves, run_pipeline)
from tests.conftest import Boom, Recorder, Touch


def state(root: str) -> str:
    return os.path.join(root, "state")


def test_sequence_passes_artifacts_by_reference(root):
    a, b = Touch("a"), Touch("b", source="@a")
    ctx = run_pipeline(Sequence("p", [a, b]), root, listeners=[])
    assert ctx["b"].path == "b.txt"
    with open(os.path.join(root, "b.txt"), encoding="utf-8") as f:
        assert f.read() == "a.txt"


def test_resume_skips_finished_steps(root):
    rec = Recorder()
    run_pipeline(Sequence("p", [Touch("a"), Touch("b", "@a")]), root, state(root), listeners=[])
    a, b = Touch("a"), Touch("b", "@a")
    run_pipeline(Sequence("p", [a, b]), root, state(root), listeners=[rec])
    assert (a.calls, b.calls) == (0, 0)
    assert rec.events == [("skipped", "a"), ("skipped", "b")]


def test_rerun_remakes_the_step_and_everything_downstream(root):
    run_pipeline(Sequence("p", [Touch("a"), Touch("b", "@a"), Touch("c")]), root, state(root), listeners=[])
    a, b, c = Touch("a"), Touch("b", "@a"), Touch("c")
    ctx = run_pipeline(Sequence("p", [a, b, c]), root, state(root), listeners=[], rerun=("a",))
    assert (a.calls, b.calls, c.calls) == (1, 1, 0)
    assert {"a", "b"} <= ctx.fresh and "b" in ctx.forced


def test_missing_output_file_runs_the_step_again(root):
    run_pipeline(Touch("a"), root, state(root), listeners=[])
    os.remove(os.path.join(root, "a.txt"))
    a = Touch("a")
    run_pipeline(a, root, state(root), listeners=[])
    assert a.calls == 1


def test_record_kinds_restore_without_a_file(root):
    run_pipeline(Touch("r", kind="report"), root, state(root), listeners=[])
    os.remove(os.path.join(root, "r.txt"))
    r = Touch("r", kind="report")
    run_pipeline(r, root, state(root), listeners=[])
    assert r.calls == 0
    assert not Artifact("report", "x").is_file and Artifact("checkpoint", "x").is_file


def test_missing_input_is_an_error(root):
    with pytest.raises(KeyError, match="needs"):
        run_pipeline(Touch("b", "@nowhere"), root, listeners=[])


def test_failure_is_reported_and_raised_and_not_recorded(root):
    rec = Recorder()
    with pytest.raises(ValueError):
        run_pipeline(Sequence("p", [Touch("a"), Boom("x")]), root, state(root), listeners=[rec])
    assert rec.events[-1] == ("failed", "x")
    assert set(Manifest(os.path.join(state(root), "manifest.json")).done) == {"a"}


def test_dry_run_records_nothing(root):
    sd = os.path.join(root, "dry")
    ctx = run_pipeline(Touch("a"), root, sd, dry_run=True, listeners=[])
    assert "a" in ctx and not os.path.exists(sd)


def test_parallel_runs_every_child(root):
    kids = [Touch(f"k{i}") for i in range(4)]
    ctx = run_pipeline(Parallel("par", list(kids)), root, state(root), listeners=[])
    assert all(k.calls == 1 for k in kids) and all(f"k{i}" in ctx for i in range(4))
    assert Parallel("empty", []).workers == 1


def test_leaves_flattens_composites():
    a, b, c = Touch("a"), Touch("b"), Touch("c")
    tree = Sequence("p", [a, Parallel("q", [b, Sequence("r", [c])])])
    assert leaves(tree) == [a, b, c]
    assert tree.describe() == "a -> q" and tree.steps[1].describe() == "b | r"


def test_context_paths_and_errors(root):
    ctx = Context(root)
    assert ctx.path("x") == os.path.join(root, "x") and ctx.path("/abs") == "/abs"
    with pytest.raises(KeyError, match="no artifact"):
        _ = ctx["missing"]


def test_manifest_round_trip(root):
    path = os.path.join(root, "m.json")
    m = Manifest(path)
    step = Touch("a")
    m.record(step, {"a": Artifact("file", "a.txt", {"x": 1.0}, {"k": "v"})})
    restored = Manifest(path).restore(step)
    assert restored == {"a": Artifact("file", "a.txt", {"x": 1.0}, {"k": "v"})}
    m.forget("a")
    assert m.restore(step) is None


def test_json_log_listener_writes_one_line_per_event(root):
    path = os.path.join(root, "events.jsonl")
    run_pipeline(Touch("a"), root, listeners=[JsonLogListener(path)])
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line)["event"] for line in f]
    assert events == ["started", "finished"]


def test_console_listener(root, capsys):
    run_pipeline(Touch("a"), root, state(root), listeners=[ConsoleListener()])
    run_pipeline(Touch("a"), root, state(root), listeners=[ConsoleListener("[x]")])
    with pytest.raises(ValueError):
        run_pipeline(Boom("b"), root, listeners=[ConsoleListener()])
    out = capsys.readouterr().out
    assert "[pipeline] > a: Touch" in out and "a = a.txt (calls 1)" in out
    assert "[x] = a: done in an earlier run" in out and "[pipeline] ! b: boom" in out


def test_composite_execute_runs_children_without_a_manifest(root):
    a = Touch("a")
    assert set(Sequence("p", [a]).execute(Context(root))) == {"a"} and a.calls == 1


def test_format_metrics():
    assert format_metrics({"a": 0.123456, "n": 3}) == "(a 0.123, n 3)"


def test_until_stops_after_a_step_and_resumes_later(root):
    a, b, c = Touch("a"), Touch("b", "@a"), Touch("c")
    ctx = run_pipeline(Sequence("p", [a, b, c]), root, state(root), listeners=[], until="b")
    assert "b" in ctx and "c" not in ctx and c.calls == 0
    a2, b2, c2 = Touch("a"), Touch("b", "@a"), Touch("c")
    run_pipeline(Sequence("p", [a2, b2, c2]), root, state(root), listeners=[], until="a")  # stops on a restored step
    assert (a2.calls, b2.calls, c2.calls) == (0, 0, 0)
    run_pipeline(Sequence("p", [a2, b2, c2]), root, state(root), listeners=[])
    assert (b2.calls, c2.calls) == (0, 1)
    with pytest.raises(KeyError, match="until"):
        run_pipeline(Touch("a"), root, listeners=[], until="nope")
