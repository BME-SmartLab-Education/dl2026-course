"""Score predictions and break the error down.

RMSE is the competition metric, but a single number will not tell you what to
fix. Slice the error by document family, language, font, font size and
photographer to see where your model actually fails, and leave `eval_ok = 0`
rows out of every number you report.

Two runs that differ by a few thousandths may not differ at all. Resample by
page, not by image, when you estimate the uncertainty of a comparison:
photographs of the same page are not independent samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y, pred) -> float:
    """Root mean squared error, the competition metric."""
    raise NotImplementedError


def by_slice(df: pd.DataFrame, pred: np.ndarray, column: str) -> pd.DataFrame:
    """RMSE per value of `column`, with the count and the mean target and prediction."""
    raise NotImplementedError


def worst(df: pd.DataFrame, pred: np.ndarray, k: int = 5) -> pd.DataFrame:
    """The k rows with the largest absolute error, with the columns worth looking at."""
    raise NotImplementedError
