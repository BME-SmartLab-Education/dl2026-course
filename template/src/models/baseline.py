"""The non-deep baseline: handcrafted features plus a classical regressor.

Its job is to establish the score a network has to beat to have earned its
complexity, and because it trains in seconds it is also the right instrument
for cheap experiments: which features matter, what a wrong split costs. The
instructor's leaderboard baseline is a model of this kind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def fit_baseline(X: pd.DataFrame, y: np.ndarray, seed: int = config.SEED):
    """Fit a regressor on the feature table, for example gradient boosting, and return it."""
    raise NotImplementedError


def predict_baseline(model, X: pd.DataFrame) -> np.ndarray:
    """Predictions clipped to [0, 1]."""
    raise NotImplementedError


def importance(model, X: pd.DataFrame, y: np.ndarray, seed: int = config.SEED) -> pd.Series:
    """How much the RMSE rises when one column is shuffled, measured on held-out rows."""
    raise NotImplementedError
