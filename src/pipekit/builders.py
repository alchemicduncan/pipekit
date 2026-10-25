"""
Building step trees: a Stage is a named group of steps a subclass builds; a Recipe is a whole pipeline, registered
under a key so a command line can find it by name.

Both are Template Methods over Sequence: the subclass sets its own attributes, then calls super().__init__(), which
calls build() / stages() to make the children.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pipekit.core import Sequence, Step


class Stage(Sequence):
    """One phase of a pipeline: a named group of steps.

    Subclasses set their own attributes BEFORE calling super().__init__(name), since build() reads them."""

    def __init__(self, name: str) -> None:
        super().__init__(name, list(self.build()))

    def build(self) -> list[Step]:
        """The stage's steps, in order."""
        raise NotImplementedError


class Recipe(Sequence):
    """A whole pipeline, named <key>[<tag>]: its stages(**options) as one step tree.

    `class MyPipeline(Recipe, key="mine")` registers it in `registry` (the nearest one up the class tree, so a project
    can keep its own: `class ProjectRecipe(Recipe): registry = {}`). Subclasses set anything stages() needs before
    calling super().__init__(tag, **options)."""

    key: ClassVar[str] = ""
    registry: ClassVar[dict[str, type[Recipe]]] = {}

    def __init_subclass__(cls, key: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if key:
            if key in cls.registry and cls.registry[key] is not cls:
                raise ValueError(f"recipe key {key!r} is taken by {cls.registry[key].__name__}")
            cls.key = key
            cls.registry[key] = cls

    def __init__(self, tag: str, **options: Any) -> None:
        self.tag = tag
        super().__init__(f"{self.key}[{tag}]", list(self.stages(**options)))

    def stages(self, **options: Any) -> list[Step]:
        """The pipeline's stages, in order."""
        raise NotImplementedError
