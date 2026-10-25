"""
pipekit: resumable pipelines of named steps - data, builds, evaluations, model training, anything with steps.

    core       Artifact, Context, Step, Sequence, Parallel, Manifest (resume), listeners, run_pipeline
    refs       "@name" references between steps
    builders   Stage and Recipe (a registered whole pipeline)
    steps      Select (best candidate by an Evaluator), Require (a metric gate)
    evaluate   Evaluator, ScoreBlend
    overrides  per-run step settings from <state_dir>/overrides.json
    process    run_logged: a subprocess with its output in a log
    cli        list/run over a recipe registry
    experiment size profiles; recipes and stages that run at several scales
    jobs       Job: an external command writing a file or folder, resumable (any kind of work)
    ml         TrainRun (a Job writing checkpoints) and picking a run's best checkpoint
"""

from pipekit.builders import Recipe, Stage
from pipekit.core import (Artifact, Composite, ConsoleListener, Context, JsonLogListener, Listener, Manifest, Parallel,
                          Sequence, Step, StopAfter, StopPipeline, leaves, run_pipeline)
from pipekit.evaluate import Evaluator, ScoreBlend
from pipekit.refs import refs, resolve
from pipekit.steps import Require, Select

__version__ = "0.1.0"

__all__ = [
    "Artifact", "Composite", "ConsoleListener", "Context", "Evaluator", "JsonLogListener", "Listener", "Manifest",
    "Parallel", "Recipe", "Require", "ScoreBlend", "Select", "Sequence", "Stage", "Step", "StopAfter", "StopPipeline",
    "leaves", "refs", "resolve",
    "run_pipeline",
]
