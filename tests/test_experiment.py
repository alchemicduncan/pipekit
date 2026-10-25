"""experiment: size profiles, experiment recipes and stages, the command line's size options."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from pipekit.builders import Recipe
from pipekit.cli import run_cli
from pipekit.core import Composite, Step
from pipekit.experiment import ExperimentRecipe, ExperimentStage, SizeProfiles
from tests.conftest import Touch

PROFILES = SizeProfiles({"quick": {"rows": 10}, "medium": {"rows": 100}, "full": {"rows": 1000}})


def test_pick():
    assert PROFILES.pick()["rows"] == 1000 and PROFILES.pick(quick=True)["rows"] == 10
    assert PROFILES.pick("medium", quick=True)["rows"] == 100 and PROFILES.names == ["quick", "medium", "full"]
    PROFILES.pick()["rows"] = 0
    assert PROFILES.pick()["rows"] == 1000  # a copy
    with pytest.raises(KeyError, match="no size profile"):
        PROFILES.pick("huge")


def test_profiles_must_match():
    with pytest.raises(ValueError, match="same keys"):
        SizeProfiles({"quick": {"a": 1}, "full": {"a": 1, "b": 2}})
    with pytest.raises(ValueError, match="need"):
        SizeProfiles({"full": {"a": 1}})


class Sample(ExperimentStage):
    """One Touch named after the experiment and its size."""

    def build(self) -> list[Step]:
        return [Touch(self.run_name(f"rows{self.size['rows']}"))]


class Experiments(ExperimentRecipe):  # pylint: disable=abstract-method
    """This module's registry, at PROFILES."""

    registry: ClassVar[dict[str, type[Recipe]]] = {}
    profiles = PROFILES


class Tiny(Experiments, key="tiny"):
    """A one-stage experiment."""

    def stages(self, **options: Any) -> list[Step]:
        return [Sample("sample", self.tag, self.size)]


def first_leaf_name(recipe: Recipe) -> str:
    stage = recipe.steps[0]
    assert isinstance(stage, Composite)
    return stage.steps[0].name


def test_experiment_recipe_sizes_its_stages():
    assert first_leaf_name(Tiny("e1")) == "e1_rows1000"
    assert first_leaf_name(Tiny("e1", quick=True)) == "e1_rows10"
    assert Tiny("e1", size_name="medium").size == {"rows": 100}


def test_recipe_without_profiles():
    class Bare(ExperimentRecipe):  # pylint: disable=abstract-method
        """No profiles."""

    with pytest.raises(TypeError, match="no size profiles"):
        Bare("t")


def test_cli_size_options(tmp_path, capsys):
    root = str(tmp_path)
    assert run_cli(Experiments.registry, ["--root", root, "run", "tiny", "--tag", "c", "--quick", "--dry-run"]) == 0
    assert run_cli(Experiments.registry, ["--root", root, "run", "tiny", "--tag", "c", "--size", "medium",
                                          "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "c_rows10" in out and "c_rows100" in out
