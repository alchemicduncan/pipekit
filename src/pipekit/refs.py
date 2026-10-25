"""
References between steps: a step's argument "@name" stands for the path of the artifact `name`, made by an earlier
step. A step declares refs(...) of its arguments as its inputs and resolve()s them when it executes.
"""

from __future__ import annotations

from typing import TypeVar

from pipekit.core import Context

T = TypeVar("T")

PREFIX = "@"


def is_ref(value: object) -> bool:
    """Whether `value` is an artifact reference ("@name")."""
    return isinstance(value, str) and value.startswith(PREFIX)


def refs(*values: object) -> tuple[str, ...]:
    """The artifact names referenced among `values` (the rest ignored): a step's inputs."""
    return tuple(v[len(PREFIX):] for v in values if isinstance(v, str) and v.startswith(PREFIX))


def resolve(ctx: Context, value: T) -> T | str:
    """'@name' -> that artifact's path; anything else unchanged."""
    if isinstance(value, str) and value.startswith(PREFIX):
        return ctx[value[len(PREFIX):]].path
    return value
