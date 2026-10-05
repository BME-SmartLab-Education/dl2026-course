"""Loading photographs and their labels.

    python -m src.data.dataset                # builds data/work/train_768 and the 64 px pixel table

Three things belong here:

* `load_table` reads `train.csv` and adds the path of every photograph;
* `build_cache` resizes every photograph once to a fixed long side, so that
  nothing downstream decodes 2048 px JPEGs every epoch;
* `pixel_table` turns every photograph into one row of grey levels, the input
  of a first, fully connected network.

Does not belong here: anything that decides which rows are training and which
are validation. That lives in splits.py, and keeping it separate is what stops
a split leaking by accident.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

RAW = config.ROOT / "data" / "raw"
WORK = config.ROOT / "data" / "work"

# These describe the page and the photographer. Use them to split and to slice
# the error, never as model input: a test image comes with none of them.
META_COLUMNS = ["page_id", "layout_id", "family", "lang", "font", "size_pt", "photographer"]


def load_table(path: Path = RAW / "train.csv") -> pd.DataFrame:
    """The label table with one extra column, `path`, pointing at `data/raw/train/<image_id>.jpg`."""
    raise NotImplementedError


def cache_dir(size: int) -> Path:
    """Where the copies with long side `size` live, for example `data/work/train_768/`."""
    raise NotImplementedError


def build_cache(df: pd.DataFrame, size: int, workers: int = 4) -> Path:
    """Resize every photograph once so that its long side is `size` px; return the folder.

    Make it idempotent (skip images already in the cache), so it is safe to
    call at the top of every script. Keep the aspect ratio. `PIL.Image.draft`
    decodes a JPEG at a fraction of its size straight from the file, which is
    several times faster than decoding at full size and resizing.
    """
    raise NotImplementedError


def pixels_of_image(path: str | Path, size: int) -> np.ndarray:
    """One photograph as a flat vector of size * size grey levels in [0, 1].

    Scale the long side to `size`, pad the short side to a square, flatten.
    Prediction must call exactly this function, so that a test image is
    preprocessed the way the training images were.
    """
    raise NotImplementedError


def pixel_table(df: pd.DataFrame, size: int = 64, workers: int = 4) -> np.ndarray:
    """All photographs as one float32 array of shape (len(df), size * size), rows in the order of `df`.

    Compute it once and store it under `data/work/`; it is small, and every
    later run should load it instead of recomputing it.
    """
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError("build the cache and the pixel table, and print how many images each holds")


if __name__ == "__main__":
    main()
