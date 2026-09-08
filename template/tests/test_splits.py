"""The one test worth writing on day one.

The same document layout appears in many photographs, and every layout exists in
both a Hungarian and an English version. A split that puts one layout on both
sides makes every score you report optimistic, by roughly 11 % on our
measurements, and nothing about the training run looks wrong when it happens.

The test below is written but skipped. Implement `build_splits` in
`src/data/splits.py`, delete the skip, and it starts protecting you.
"""
from __future__ import annotations

import pytest


def test_imports() -> None:
    """The project is importable and the paths resolve. Fails if PYTHONPATH is wrong."""
    from src import config

    assert config.ROOT.exists()
    assert config.TARGET == "usability"


def test_no_layout_appears_in_both_splits() -> None:
    """No layout group may be in training and validation at the same time."""
    splits = pytest.importorskip("src.data.splits")
    if not hasattr(splits, "build_splits"):
        pytest.skip("write build_splits() in src/data/splits.py, then delete this skip")

    # Replace with your own label table once the data is in place.
    import pandas as pd

    df = pd.DataFrame({
        "filename": [f"img_{i}.jpg" for i in range(40)],
        "layout_id": [i // 4 for i in range(40)],
        "lang": ["hu", "en"] * 20,
        "usability": [0.5] * 40,
    })

    train, val = splits.build_splits(df, group_column="layout_id", val_fraction=0.25)

    overlap = set(df.loc[train, "layout_id"]) & set(df.loc[val, "layout_id"])
    assert not overlap, f"layouts on both sides of the split: {sorted(overlap)}"
    assert len(train) + len(val) == len(df), "the split lost or duplicated rows"
    assert len(val) > 0, "the validation set is empty"
