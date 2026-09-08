"""Experiment tracking with Weights & Biases.

Print statements vanish. A run that recorded its config, its metrics per epoch
and its final score can be compared against the twenty runs that came after it,
which is the only way to tell an improvement from noise.

SETUP
-----
    cp .env.example .env        # then paste your key from wandb.ai/authorize

Docker Compose reads `.env` automatically and passes `WANDB_API_KEY`,
`WANDB_ENTITY` and `WANDB_MODE` into the container. `.env` is gitignored:
never commit your key.

`WANDB_MODE=offline` logs to disk if you have no network; `wandb sync wandb/…`
uploads it later. `WANDB_MODE=disabled` turns it off entirely, which is useful
when you are debugging and do not want fifty junk runs in your project.

WHAT TO LOG
-----------
The resolved config, so a run is reproducible from its own record. Train and
validation loss per epoch. The final metrics. The git commit, which wandb picks
up on its own. The checkpoint, as an artifact, so the weights that produced a
number are attached to that number rather than to a filename you will not
recognise in December.

Log the *predictions* of your best run too. Comparing two runs' errors on the
same images tells you far more than comparing two RMSE values.
"""
from __future__ import annotations


def init_run(cfg: dict, tags: list[str] | None = None):
    """Start a run. Pass the resolved config so it is recorded with the metrics.

    Give every run a name you will recognise later. `resnet18_384_grouped` beats
    `run_47`.
    """
    raise NotImplementedError


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a dictionary of scalars, typically once per epoch."""
    raise NotImplementedError


def log_checkpoint(path, name: str, metadata: dict | None = None) -> None:
    """Attach a checkpoint to the run as an artifact.

    Worth doing properly: the project requires that the checkpoint you hand in is
    the one that produced your submitted predictions. If every checkpoint is
    logged against the run that made it, that question answers itself.
    """
    raise NotImplementedError


def finish() -> None:
    """Close the run. Do this even when training crashes, or the run stays
    marked as running forever."""
    raise NotImplementedError
