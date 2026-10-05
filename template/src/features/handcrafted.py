"""Classical image-quality features for the baseline model.

Things worth measuring: sharpness (Laplacian variance, Tenengrad), exposure
(mean, percentiles, saturated and dark pixel fractions), contrast, histogram
shape, JPEG blockiness, and the geometry of the page quadrilateral if you
detect it from the page edges. The corner markers have been painted out, so
they cannot be used for that.

Worth thinking about: all of those describe the frame as a whole. A defect that
covers one column and leaves the rest clean moves them very little. Tile-wise
statistics on a grid, summarised with something other than the mean, are one
way to give a classical model a chance at that.

Compute every feature on the same resized cache, so that a sharpness number
means the same thing on every row.
"""
from __future__ import annotations

import pandas as pd

FEATURE_SIZE = 768

# name -> function(gray uint8 image) -> float
FEATURES: dict = {}


def extract(image_id: str, size: int = FEATURE_SIZE) -> dict:
    """Every feature in FEATURES for one cached photograph."""
    raise NotImplementedError


def feature_table(df: pd.DataFrame, size: int = FEATURE_SIZE, workers: int = 4) -> pd.DataFrame:
    """One row of features per row of `df`, same index."""
    raise NotImplementedError


def features_of_image(path: str, size: int = FEATURE_SIZE) -> dict:
    """The same features for an arbitrary image file, for prediction on photographs not in the cache.

    Resize exactly as the cache does, or a number computed here will not mean
    what it meant in training.
    """
    raise NotImplementedError
