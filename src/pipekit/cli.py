"""
A command line over a recipe registry:

    pipekit --module myproject.recipes list
    pipekit --module myproject.recipes run mine --tag v1 --dry-run
    pipekit --module myproject.recipes run mine --tag v1 --option steps=1000 --rerun train
    pipekit --module myproject.recipes run mine --tag v1 --size quick      # an ExperimentRecipe at its quick size

State (manifest, events, overrides) lives in <state-root>/<tag>. A project can also call run_cli() from its own
entry point with its registry.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

from pipekit import overrides
from pipekit.builders import Recipe
from pipekit.core import Step, run_pipeline


def parse_option(text: str) -> tuple[str, Any]:
    """'KEY=VALUE' -> (key, value), with true/false, ints and floats converted."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"--option {text!r}: expected KEY=VALUE")
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    for kind in (int, float):
        try:
            return key, kind(value)
        except ValueError:
            pass
    return key, value


def plan(step: Step, depth: int = 0) -> list[str]:
    """The tree as indented '- name: description' lines."""
    lines = ["  " * depth + f"- {step.name}: {step.describe()}"]
    for child in getattr(step, "steps", []):
        lines += plan(child, depth + 1)
    return lines


def build_parser(registry: Mapping[str, type[Recipe]], prog: str = "pipekit") -> argparse.ArgumentParser:
    """The list/run command line for `registry`."""
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--root", default=".", help="relative artifact paths resolve against it (default: here)")
    ap.add_argument("--state-root", default="pipelines", help="state goes in <state-root>/<tag> (relative to root)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the recipes")
    run = sub.add_parser("run", help="run (or --dry-run) a recipe")
    run.add_argument("recipe", choices=sorted(registry))
    run.add_argument("--tag", required=True, help="names this pipeline's state folder")
    run.add_argument("--dry-run", action="store_true", help="walk the tree without doing the work")
    run.add_argument("--size", default=None, metavar="NAME", help="a size profile (an ExperimentRecipe's profiles)")
    run.add_argument("--quick", action="store_true", help="the quick size profile: try the recipe end to end")
    run.add_argument("--rerun", nargs="*", default=[], help="forget these finished steps and run them again")
    run.add_argument("--until", default=None, metavar="STEP", help="stop after this step (the rest runs next time)")
    run.add_argument("--option", action="append", default=[], type=parse_option, metavar="KEY=VALUE",
                     help="a recipe option (numbers and true/false are converted)")
    return ap


def run_cli(registry: Mapping[str, type[Recipe]], argv: Sequence[str] | None = None, prog: str = "pipekit") -> int:
    """Parse argv and list or run a recipe from `registry`; returns an exit code."""
    args = build_parser(registry, prog).parse_args(argv)
    if args.cmd == "list":
        for name, recipe in sorted(registry.items()):
            print(f"{name}: {(recipe.__doc__ or '').strip().splitlines()[0] if recipe.__doc__ else ''}")
        return 0
    root = os.path.abspath(args.root)
    sizing: dict[str, Any] = ({"size_name": args.size} if args.size else {}) | ({"quick": True} if args.quick else {})
    pipeline = registry[args.recipe](tag=args.tag, **sizing, **dict(args.option))
    state_dir = os.path.join(root, args.state_root, args.tag)
    for change in overrides.apply(pipeline, overrides.load(state_dir)):
        print(f"override {change}")
    print("\n".join(plan(pipeline)))
    run_pipeline(pipeline, root=root, state_dir=state_dir, dry_run=args.dry_run, rerun=tuple(args.rerun),
                 until=args.until)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """`pipekit --module pkg.recipes ...`: import the module (its Recipe subclasses register), then run_cli."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--module", action="append", default=[], help="import this module for its recipes (repeatable)")
    known, rest = pre.parse_known_args(argv)
    for module in known.module:
        importlib.import_module(module)
    return run_cli(Recipe.registry, rest)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
