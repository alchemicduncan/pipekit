"""
Experiments at several scales: the same pipeline tried end to end small, then run for real.

  SizeProfiles      named budgets ("quick", "medium", "full"): the same keys (step counts, row limits, episodes,
                    workers, ...) in each, so a recipe tried at "quick" runs the very same tree at "full"
  ExperimentRecipe  a Recipe that picks its size profile when built (quick= / size_name=) and hands it to stages
  ExperimentStage   a Stage that knows the experiment's tag and size; names what it makes <tag>_<suffix>
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from pipekit.builders import Recipe, Stage

Size = dict[str, Any]


class SizeProfiles:
    """Named budgets. Every profile must have the same keys, so no stage can work at one size and fail at another for
    want of a setting."""

    def __init__(self, profiles: Mapping[str, Mapping[str, Any]], default: str = "full", quick: str = "quick") -> None:
        if default not in profiles or quick not in profiles:
            raise ValueError(f"profiles need {default!r} and {quick!r} (have: {', '.join(profiles)})")
        keys = set().union(*(set(p) for p in profiles.values()))
        uneven = {name: sorted(keys - set(p)) for name, p in profiles.items() if keys - set(p)}
        if uneven:
            raise ValueError(f"every profile needs the same keys; missing: {uneven}")
        self.profiles = {name: dict(p) for name, p in profiles.items()}
        self.default, self.quick = default, quick

    @property
    def names(self) -> list[str]:
        """The profiles, in the order given."""
        return list(self.profiles)

    def pick(self, name: str | None = None, quick: bool = False) -> Size:
        """The profile `name`, else the quick one if `quick`, else the default (a copy: a recipe may change it)."""
        chosen = name or (self.quick if quick else self.default)
        if chosen not in self.profiles:
            raise KeyError(f"no size profile {chosen!r} (have: {', '.join(self.profiles)})")
        return dict(self.profiles[chosen])


class ExperimentRecipe(Recipe):  # pylint: disable=abstract-method
    """A Recipe at a size: `profiles` (a class attribute) supplies the budgets; self.size is the one picked, set before
    stages() runs. `class MyRecipe(ExperimentRecipe): profiles = SizeProfiles({...})`."""

    profiles: ClassVar[SizeProfiles | None] = None

    def __init__(self, tag: str, quick: bool = False, size_name: str | None = None, **options: Any) -> None:
        if self.profiles is None:
            raise TypeError(f"{type(self).__name__} has no size profiles (set `profiles = SizeProfiles(...)`)")
        self.quick = quick
        self.size = self.profiles.pick(size_name, quick)
        super().__init__(tag, **options)


class ExperimentStage(Stage):  # pylint: disable=abstract-method
    """A Stage of an experiment: the experiment's tag and size are set before build() runs.

    Subclasses set their own attributes BEFORE calling super().__init__(name, tag, size)."""

    def __init__(self, name: str, tag: str, size: Size) -> None:
        self.tag, self.size = tag, size
        super().__init__(name)

    def run_name(self, suffix: str) -> str:
        """A name for something this stage makes (a run, a dataset, a report), the experiment's tag first, so
        everything of one experiment sorts together."""
        return f"{self.tag}_{suffix}"
