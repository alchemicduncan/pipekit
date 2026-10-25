"""cli: options, the plan, list and run."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import pytest

from pipekit.builders import Recipe
from pipekit.cli import main, parse_option, plan, run_cli
from tests.test_parts import Demo, Projects


@pytest.mark.parametrize("text, expected", [
    ("a=1", ("a", 1)), ("a=1.5", ("a", 1.5)), ("a=TRUE", ("a", True)), ("a=false", ("a", False)),
    ("a=x=y", ("a", "x=y")), ("a=", ("a", "")),
])
def test_parse_option(text, expected):
    assert parse_option(text) == expected


@pytest.mark.parametrize("text", ["novalue", "=1"])
def test_parse_option_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_option(text)


def test_plan_indents_the_tree():
    assert plan(Demo("t")) == ["- demo[t]: pair", "  - pair: x -> x2", "    - x: Touch", "    - x2: Touch"]


def test_list(capsys):
    assert run_cli(Projects.registry, ["list"]) == 0
    assert capsys.readouterr().out.strip() == "demo: A demo pipeline."


def test_run_applies_options_and_overrides_and_resumes(tmp_path, capsys):
    root = str(tmp_path)
    sd = os.path.join(root, "pipelines", "t1")
    os.makedirs(sd)
    with open(os.path.join(sd, "overrides.json"), "w", encoding="utf-8") as f:
        json.dump({"z2": {"kind": "report"}}, f)
    args = ["--root", root, "run", "demo", "--tag", "t1", "--option", "first=z"]
    assert run_cli(Projects.registry, args) == 0
    assert os.path.exists(os.path.join(root, "z2.txt"))
    assert run_cli(Projects.registry, args + ["--rerun", "z2"]) == 0
    out = capsys.readouterr().out
    assert "override z2.kind: 'file' -> 'report'" in out
    assert "= z: done in an earlier run" in out and "> z2: Touch" in out.split("override")[-1]


def test_dry_run_leaves_no_state(tmp_path):
    root = str(tmp_path)
    assert run_cli(Projects.registry, ["--root", root, "run", "demo", "--tag", "t2", "--dry-run"]) == 0
    assert not os.path.exists(os.path.join(root, "pipelines", "t2"))


def test_main_imports_recipe_modules(tmp_path, monkeypatch, capsys):
    module = tmp_path / "pk_demo_recipes.py"
    module.write_text(textwrap.dedent('''
        from pipekit.builders import Recipe
        class Hello(Recipe, key="pk_hello"):
            """Says hello."""
            def stages(self, **options):
                return []
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert main(["--module", "pk_demo_recipes", "list"]) == 0
        assert "pk_hello: Says hello." in capsys.readouterr().out
    finally:
        Recipe.registry.pop("pk_hello", None)
        sys.modules.pop("pk_demo_recipes", None)


def test_run_until(tmp_path, capsys):
    root = str(tmp_path)
    assert run_cli(Projects.registry, ["--root", root, "run", "demo", "--tag", "u", "--until", "x"]) == 0
    assert os.path.exists(os.path.join(root, "x.txt")) and not os.path.exists(os.path.join(root, "x2.txt"))
    assert "[pipeline] > x2" not in capsys.readouterr().out
