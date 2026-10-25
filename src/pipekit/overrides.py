"""
Experiment tweaks without code edits: <state_dir>/overrides.json names steps and the attributes to change on them,
applied to the tree after it is built.

    {"train": {"ppo": {"lr": 1e-4}, "timesteps": 12000000},
     "gate": {"maximum": 0.3}}

A dict value merges into a dict attribute; anything else replaces it. A step that already ran with other settings is
not rerun by this alone (rerun it explicitly). Unknown steps or attributes are errors, so a typo cannot silently
change nothing.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pipekit.core import Step, leaves

FILENAME = "overrides.json"

Overrides = dict[str, dict[str, Any]]


def path_for(state_dir: str) -> str:
    """Where a run's overrides live."""
    return os.path.join(state_dir, FILENAME)


def load(state_dir: str) -> Overrides:
    """The run's overrides, or none."""
    path = path_for(state_dir)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data: Overrides = json.load(f)
    return data


def save(state_dir: str, overrides: Overrides) -> None:
    """Write the run's overrides (atomically)."""
    os.makedirs(state_dir, exist_ok=True)
    tmp = path_for(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=1, sort_keys=True)
    os.replace(tmp, path_for(state_dir))


def apply(tree: Step, overrides: Overrides) -> list[str]:
    """Set each named step's attributes (a visit over the tree's leaves); returns what changed, for the log."""
    steps = {s.name: s for s in leaves(tree)}
    changed = []
    for name, attrs in overrides.items():
        if name not in steps:
            raise KeyError(f"overrides: no step {name!r} in this pipeline (have: {', '.join(sorted(steps))})")
        step = steps[name]
        for attr, value in attrs.items():
            if not hasattr(step, attr):
                raise AttributeError(f"overrides: step {name!r} ({type(step).__name__}) has no attribute {attr!r}")
            old = getattr(step, attr)
            new = {**old, **value} if isinstance(old, dict) and isinstance(value, dict) else value
            setattr(step, attr, new)
            changed.append(f"{name}.{attr}: {old!r} -> {new!r}")
    return changed
