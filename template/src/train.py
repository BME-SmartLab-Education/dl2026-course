"""Train a model from a config and log the run to Weights & Biases.

    python -m src.train --config configs/mlp.yaml

One entry point, driven by a config, so that a result can be traced back to
the settings that produced it. The pieces, each with its contract below:

    make_optimizer   the optimiser from the config
    train_epoch      one pass over the training rows in shuffled minibatches
    predict          model outputs as numpy, in evaluation mode
    validate         RMSE on the scoreable validation rows
    train_model      the epoch loop: train, validate, log, checkpoint

Standardise the inputs with statistics from the training rows only, and store
them in the checkpoint: prediction has to use exactly the same transform.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """The optimiser, using cfg['train']['lr'] and cfg['train']['weight_decay']."""
    raise NotImplementedError


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator) -> float:
    """One pass over the training rows in shuffled minibatches; return the mean training loss.

    Draw the order from `generator` so the run is reproducible, and step the
    optimiser once per minibatch.
    """
    raise NotImplementedError


def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """The model's outputs for every row of X, as a numpy array of shape (len(X),).

    Evaluation mode, no gradients, and the same answer for a row whether it
    is predicted alone or inside a batch.
    """
    raise NotImplementedError


def validate(model: torch.nn.Module, X_val: torch.Tensor, y_val: np.ndarray, ok: np.ndarray) -> float:
    """RMSE on the validation rows where `ok` (eval_ok == 1) is True."""
    raise NotImplementedError


def train_model(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    """Train on train_idx, validate on val_idx after every epoch, log to W&B, save the checkpoint.

    Log the resolved config, the training and validation metrics per epoch,
    and the checkpoint as an artifact (see src/tracking.py), so that the
    weights behind a number stay attached to that number.
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a model from a config")
    ap.add_argument("--config", required=True)
    ap.parse_args()
    raise NotImplementedError("load the config, the data and the split, then call train_model")


if __name__ == "__main__":
    main()
