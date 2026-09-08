"""Project-wide settings.

Keep every path and hyperparameter that more than one script needs in here, so
an experiment is described by a config rather than by whatever was typed on the
command line at the time.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"          # training logs, metrics, TensorBoard, figures
SUBMISSIONS = ROOT / "submissions"

# The target is continuous on [0, 1]; the competition metric is RMSE.
TARGET = "usability"
SEED = 42
