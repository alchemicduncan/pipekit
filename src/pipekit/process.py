"""Running a step's work as a subprocess, its output appended to a log file."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence


class CommandFailed(RuntimeError):
    """A logged command exited non-zero."""

    def __init__(self, args: Sequence[str], returncode: int, log_path: str) -> None:
        self.args_run, self.returncode, self.log_path = list(args), returncode, log_path
        name = os.path.basename(args[1]) if len(args) > 1 else args[0]
        super().__init__(f"{name} failed (exit {returncode}); see {log_path}")


def run_logged(args: Sequence[str], log_path: str, env: Mapping[str, str] | None = None,
               cwd: str | None = None) -> None:
    """Run a command to completion with its output appended to log_path (after a '$ command' line); raise
    CommandFailed if it exits non-zero. `env` is added to this process's environment."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(args) + "\n")
        log.flush()
        result = subprocess.run(list(args), cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})}, check=False)
    if result.returncode != 0:
        raise CommandFailed(args, result.returncode, log_path)
