# pipekit

Resumable pipelines of named steps. Each step declares the artifacts it reads and writes. A manifest records the
steps that have finished, so a re-run skips them, and a step whose inputs were remade runs again along with
everything downstream of it. Listeners report progress, so steps never print anything themselves.

Any work that comes in steps fits: data preparation, builds, evaluations, reports, model training. pipekit holds the
general machinery; everything specific to one project (its simulators, datasets, evaluations) stays in that project,
in steps and recipes built on these classes.

| Module | What it holds |
|---|---|
| `pipekit.core` | `Artifact`, `Context`, `Step`, `Sequence`, `Parallel`, `Manifest`, `Listener`s, `leaves`, `run_pipeline` |
| `pipekit.refs` | `"@name"` references between steps: `refs()` for a step's inputs, `resolve()` when it executes |
| `pipekit.builders` | `Stage` (a group of steps) and `Recipe` (a whole pipeline, registered by key) |
| `pipekit.steps` | `Select` (the best candidate by an `Evaluator`) and `Require` (a metric gate) |
| `pipekit.evaluate` | `Evaluator` and `ScoreBlend` |
| `pipekit.overrides` | Per-run step settings from `<state_dir>/overrides.json` |
| `pipekit.process` | `run_logged`: runs a subprocess with its output appended to a log |
| `pipekit.cli` | `list` and `run` over a recipe registry (with `--size` / `--quick` for experiment recipes) |
| `pipekit.experiment` | `SizeProfiles`, `ExperimentRecipe`, `ExperimentStage`: one pipeline at several scales (quick, full) |
| `pipekit.jobs` | `Job`: an external command writing a file or folder, resumable, set aside on rerun (`set_aside`) |
| `pipekit.ml` | `TrainRun` (a `Job` writing checkpoints), `checkpoints_in`, `checkpoint_candidates` (for `Select`) |

## Example

```python
from pipekit import Artifact, Recipe, Require, Step, refs, resolve

class Train(Step):
    def __init__(self, name, data):
        super().__init__(name)
        self.data, self.inputs, self.outputs = data, refs(data), (name,)

    def execute(self, ctx):
        path = f"models/{self.name}.pt"   # ... train on resolve(ctx, self.data) ...
        return {self.name: Artifact("checkpoint", path, metrics={"loss": 0.1})}

class Demo(Recipe, key="demo"):
    """Prepare, train, gate."""
    def stages(self, data="data.csv"):
        return [Train("model", data), Require("model.ok", "model", "loss", maximum=0.5)]
```

```
pipekit --module myproject.recipes run demo --tag v1 --dry-run
pipekit --module myproject.recipes run demo --tag v1 --option data=other.csv
pipekit --module myproject.recipes run demo --tag v1 --rerun model
```

State is kept in `pipelines/<tag>/`: `manifest.json`, `events.jsonl` and `overrides.json`.

## External commands: `Job`

```python
import sys
from pipekit import Recipe, Require
from pipekit.jobs import Job

class Export(Job):
    """Writes data/rows.csv; done once, set aside and redone on --rerun export."""
    def command(self, ctx):
        return [sys.executable, "export.py", "--out", self.out]

class Report(Job):
    kind = "report"
    def command(self, ctx):
        return [sys.executable, "report.py", "--rows", "data/rows.csv", "--out", self.out]

class Nightly(Recipe, key="nightly"):
    """Export, then report."""
    def stages(self):
        return [Export("export", "data/rows.csv"), Report("report", "reports/today.json")]
```

A file job keeps its log and done marker beside its output (`data/rows.csv.log`, `data/rows.csv.done`); a job with
`folder = True` keeps them inside it.

## An experiment at several sizes, with training

```python
import sys
from pipekit import Select
from pipekit.experiment import ExperimentRecipe, ExperimentStage, SizeProfiles
from pipekit.ml import TrainRun, checkpoint_candidates

class Fit(TrainRun):
    """One training job: its run folder holds the checkpoints."""
    def __init__(self, name, run_dir, steps):
        super().__init__(name, run_dir)
        self.steps = steps

    def command(self, ctx):
        return [sys.executable, "train.py", "--steps", str(self.steps), "--out", self.run_dir]

class Learn(ExperimentStage):
    def build(self):
        run = self.run_name("fit")                    # <tag>_fit
        return [Fit("train", f"runs/{run}", self.size["steps"]),
                Select("best", "@train.run", MyEvaluator(self.size["eval_rows"]), checkpoint_candidates())]

class Classifier(ExperimentRecipe, key="classifier"):
    """Train, then keep the best checkpoint."""
    profiles = SizeProfiles({"quick": {"steps": 100, "eval_rows": 50},
                             "full": {"steps": 50_000, "eval_rows": 10_000}})
    def stages(self):
        return [Learn("learn", self.tag, self.size)]
```

```
pipekit --module myproject.recipes run classifier --tag v1 --quick    # the whole tree, end to end, small
pipekit --module myproject.recipes run classifier --tag v1            # the same tree at full size
```

## Development

```
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest --cov=pipekit
.venv/bin/mypy
.venv/bin/pylint src tests
```
