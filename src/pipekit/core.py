"""
The building blocks: artifacts, the context steps share, steps, and composites of steps.

Patterns, and what each is for:

  Command + Template Method  Step.run() is the fixed skeleton - skip if the manifest says this step is done, check
                             its inputs exist, execute(), record its outputs - and each concrete step only supplies
                             execute(). A step is an object holding everything needed to run later.
  Composite                  Sequence and Parallel are Steps made of Steps, so a whole chain nests inside a bigger
                             one and runs with the same call.
  Memento                    Manifest saves each finished step's outputs to JSON; a re-run restores them and skips
                             the step, so a pipeline resumes where it stopped (or a crash left it). A step whose
                             inputs were remade in this run runs again, and so on down the chain.
  Observer                   Listeners hear every step start, finish, skip or fail (console, JSON log), so the
                             steps themselves never print or log.

Artifacts flow by name: a step declares the names it reads (`inputs`) and the names it writes (`outputs`); the
Context is the blackboard between them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

# artifact kinds that are records rather than files: restoring them from the manifest needs nothing on disk
RECORD_KINDS = frozenset({"report"})


@dataclass
class Artifact:
    """Something a step made: a checkpoint, an export, an evaluation report, ..."""

    kind: str
    path: str
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_file(self) -> bool:
        """Whether `path` names something on disk (a record kind does not)."""
        return self.kind not in RECORD_KINDS


class Context:
    """The blackboard steps read their inputs from and write their outputs to, by name."""

    def __init__(self, root: str, dry_run: bool = False) -> None:
        self.root = root  # relative artifact paths resolve against it
        self.dry_run = dry_run
        self.artifacts: dict[str, Artifact] = {}
        self.fresh: set[str] = set()  # artifacts made (not restored) in this run: whatever reads them is stale
        self.forced: set[str] = set()  # steps that must not adopt an earlier attempt's output (rerun, or stale inputs)

    def __getitem__(self, name: str) -> Artifact:
        if name not in self.artifacts:
            raise KeyError(f"no artifact {name!r} yet (have: {', '.join(sorted(self.artifacts)) or 'none'})")
        return self.artifacts[name]

    def __contains__(self, name: object) -> bool:
        return name in self.artifacts

    def put(self, name: str, artifact: Artifact) -> None:
        """Make `artifact` available to later steps as `name`."""
        self.artifacts[name] = artifact

    def path(self, relative: str) -> str:
        """`relative` resolved against the context's root (absolute paths are left alone)."""
        return relative if os.path.isabs(relative) else os.path.join(self.root, relative)


class Listener:
    """Observer of a pipeline run. Override what you need."""

    def step_started(self, step: Step) -> None:
        """A step is about to execute."""

    def step_finished(self, step: Step, outputs: dict[str, Artifact], seconds: float) -> None:
        """A step executed and made `outputs`."""

    def step_skipped(self, step: Step, reason: str) -> None:
        """A step was not executed (its outputs were restored)."""

    def step_failed(self, step: Step, error: BaseException) -> None:
        """A step raised `error`."""


def format_metrics(metrics: dict[str, Any]) -> str:
    """'(name value, ...)' with floats to 3 significant figures."""
    return "(" + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()) + ")"


class ConsoleListener(Listener):
    """One line per event on stdout: '>' started, '<' finished, '=' skipped, '!' failed."""

    def __init__(self, prefix: str = "[pipeline]") -> None:
        self.prefix = prefix

    def step_started(self, step: Step) -> None:
        print(f"{self.prefix} > {step.name}: {step.describe()}", flush=True)

    def step_finished(self, step: Step, outputs: dict[str, Artifact], seconds: float) -> None:
        made = ", ".join(f"{k} = {v.path}" + (f" {format_metrics(v.metrics)}" if v.metrics else "")
                         for k, v in outputs.items())
        print(f"{self.prefix} < {step.name} ({seconds / 60:.1f} min){': ' + made if made else ''}", flush=True)

    def step_skipped(self, step: Step, reason: str) -> None:
        print(f"{self.prefix} = {step.name}: {reason}", flush=True)

    def step_failed(self, step: Step, error: BaseException) -> None:
        print(f"{self.prefix} ! {step.name}: {error}", flush=True)


class JsonLogListener(Listener):
    """One JSON line per event, appended to a file - the run's history, readable by other tools."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()  # Parallel steps report from several threads

    def _write(self, **event: Any) -> None:
        line = json.dumps({"time": time.time(), **event}, default=str) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def step_started(self, step: Step) -> None:
        self._write(event="started", step=step.name)

    def step_finished(self, step: Step, outputs: dict[str, Artifact], seconds: float) -> None:
        self._write(event="finished", step=step.name, seconds=seconds,
                    outputs={k: asdict(v) for k, v in outputs.items()})

    def step_skipped(self, step: Step, reason: str) -> None:
        self._write(event="skipped", step=step.name, reason=reason)

    def step_failed(self, step: Step, error: BaseException) -> None:
        self._write(event="failed", step=step.name, error=repr(error))


class Manifest:
    """Memento of a pipeline run: the outputs of every finished step, saved (atomically) after each one."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()  # Parallel steps finish on several threads, all writing the one file
        self.done: dict[str, dict[str, dict[str, Any]]] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.done = json.load(f)

    def restore(self, step: Step) -> dict[str, Artifact] | None:
        """The outputs `step` recorded in an earlier run, or None."""
        if step.name not in self.done:
            return None
        return {name: Artifact(**a) for name, a in self.done[step.name].items()}

    def record(self, step: Step, outputs: dict[str, Artifact]) -> None:
        """Save `step` as done with `outputs`."""
        with self._lock:
            self.done[step.name] = {name: asdict(a) for name, a in outputs.items()}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.done, f, indent=1)
            os.replace(tmp, self.path)

    def forget(self, step_name: str) -> None:
        """Drop a step's record, so the next run executes it again."""
        self.done.pop(step_name, None)


class Step(ABC):
    """One unit of work (Command). run() is the template; subclasses implement execute()."""

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __init__(self, name: str) -> None:
        self.name = name

    def describe(self) -> str:
        """One line on what the step does, for plans and logs."""
        return type(self).__name__

    def run(self, ctx: Context, manifest: Manifest | None, listeners: list[Listener]) -> dict[str, Artifact]:
        """Restore this step's outputs if an earlier run made them (and its inputs are not new), else execute it."""
        stale = [name for name in self.inputs if name in ctx.fresh]  # an input was remade this run: so is this step
        if stale and manifest is not None and manifest.restore(self) is not None:
            ctx.forced.add(self.name)
            manifest.forget(self.name)
        restored = manifest.restore(self) if manifest is not None else None
        if restored is not None and all(os.path.exists(ctx.path(a.path)) for a in restored.values() if a.is_file):
            for name, artifact in restored.items():
                ctx.put(name, artifact)
            for listener in listeners:
                listener.step_skipped(self, "done in an earlier run")
            return restored
        missing = [name for name in self.inputs if name not in ctx]
        if missing:
            raise KeyError(f"step {self.name!r} needs {missing}, which no earlier step made")
        for listener in listeners:
            listener.step_started(self)
        start = time.time()
        try:
            outputs = self.execute(ctx)
        except BaseException as error:
            for listener in listeners:
                listener.step_failed(self, error)
            raise
        for name, artifact in outputs.items():
            ctx.put(name, artifact)
        ctx.fresh.update(outputs)
        if manifest is not None and not ctx.dry_run:
            manifest.record(self, outputs)
        for listener in listeners:
            listener.step_finished(self, outputs, time.time() - start)
        return outputs

    @abstractmethod
    def execute(self, ctx: Context) -> dict[str, Artifact]:
        """Do the work; return the artifacts made, by name."""


class Composite(Step):
    """A Step made of Steps. Its run() runs the children (each skipping or recording itself); execute() runs them with
    no manifest or listeners."""

    def __init__(self, name: str, steps: list[Step]) -> None:
        super().__init__(name)
        self.steps = steps

    def execute(self, ctx: Context) -> dict[str, Artifact]:
        return self.run(ctx, None, [])


class Sequence(Composite):
    """Run steps one after another; each sees what the earlier ones made."""

    def describe(self) -> str:
        return " -> ".join(s.name for s in self.steps)

    def run(self, ctx: Context, manifest: Manifest | None, listeners: list[Listener]) -> dict[str, Artifact]:
        made: dict[str, Artifact] = {}
        for step in self.steps:
            made.update(step.run(ctx, manifest, listeners))
        return made


class Parallel(Composite):
    """Run independent steps at the same time (threads), then continue."""

    def __init__(self, name: str, steps: list[Step], workers: int | None = None) -> None:
        super().__init__(name, steps)
        self.workers = workers or max(len(steps), 1)

    def describe(self) -> str:
        return " | ".join(s.name for s in self.steps)

    def run(self, ctx: Context, manifest: Manifest | None, listeners: list[Listener]) -> dict[str, Artifact]:
        made: dict[str, Artifact] = {}
        with ThreadPoolExecutor(self.workers) as pool:
            for result in pool.map(lambda s: s.run(ctx, manifest, listeners), self.steps):
                made.update(result)
        return made


class StopPipeline(Exception):
    """Raised (by StopAfter) to end a run early and cleanly; run_pipeline catches it."""


class StopAfter(Listener):
    """Ends the run once the step named `step` has finished or been restored (e.g. train, look, then decide)."""

    def __init__(self, step: str) -> None:
        self.step = step
        self.reached = False

    def _check(self, step: Step) -> None:
        if step.name == self.step:
            self.reached = True
            raise StopPipeline(f"stopped after {step.name}")

    def step_finished(self, step: Step, outputs: dict[str, Artifact], seconds: float) -> None:
        self._check(step)

    def step_skipped(self, step: Step, reason: str) -> None:
        self._check(step)


def leaves(step: Step) -> list[Step]:
    """The steps that do the work, in order (Iterator over the Composite)."""
    if isinstance(step, Composite):
        return [leaf for kid in step.steps for leaf in leaves(kid)]
    return [step]


def run_pipeline(pipeline: Step, root: str, state_dir: str | None = None, dry_run: bool = False,
                 listeners: list[Listener] | None = None, rerun: tuple[str, ...] = (),
                 until: str | None = None) -> Context:
    """Run a step tree with a manifest (resume) and listeners. `rerun` forgets those steps first; `until` ends the run
    after that step (finished or restored), leaving the rest for a later run.

    With a state_dir, the manifest is <state_dir>/manifest.json and every event is appended to events.jsonl."""
    ctx = Context(root, dry_run=dry_run)
    manifest = None
    listeners = list(listeners) if listeners is not None else [ConsoleListener()]
    if state_dir is not None:
        if not dry_run:
            os.makedirs(state_dir, exist_ok=True)
        manifest = Manifest(os.path.join(state_dir, "manifest.json"))
        for name in rerun:
            manifest.forget(name)
        ctx.forced.update(rerun)
        if not dry_run:
            listeners.append(JsonLogListener(os.path.join(state_dir, "events.jsonl")))
    if until is not None:
        if until not in {s.name for s in leaves(pipeline)}:
            raise KeyError(f"until: no step {until!r} in {pipeline.name}")
        listeners.append(StopAfter(until))
    try:
        pipeline.run(ctx, manifest, listeners)
    except StopPipeline:
        pass
    return ctx
