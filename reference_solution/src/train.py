"""Train lab 4's network to predict usability.

    python -m src.train --fold 0      # train on folds 1-3, report RMSE on fold 0
    python -m src.train --fold all    # train on every photograph: the model you submit

A fold run writes its checkpoint and its validation predictions; the `all` run
writes only the checkpoint, since nothing is left to validate on. Both use the
same number of epochs, the one in the config, so what you measured on a fold is
what you get from the final run. Each run prints the line to put in your notes.

What one run does, in order:
  1. reads the config and fixes the random seed,
  2. reads train.csv and deals the photographs into folds by `group_strict`,
  3. builds the data loaders from the resized copies that prepare.py wrote,
  4. builds the network and measures the pixel statistics it standardises with,
  5. trains for a fixed number of epochs with AdamW and a warm-up-cosine schedule,
  6. saves the checkpoint, the per-epoch log and (fold runs only) the
     validation predictions.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common import (
    TARGET,
    Photos,
    assign_folds,
    autocast,
    build_model,
    cache_path,
    device,
    load_config,
    predict,
    rmse,
    seed_everything,
    train_table,
    worker_init,
)


def lr_at(step: int, total: int, warmup: float) -> float:
    """Linear warm-up, then a cosine down to zero; a factor on the base learning rate.

    `step` counts optimiser steps (batches), not epochs. The first `warmup`
    fraction of them ramps the factor from almost 0 to 1, so that the first,
    still random updates cannot throw the network far off; the rest follows
    half a cosine from 1 to 0, so that the last epoch makes only small changes
    and the final weights are a settled point rather than a random one.
    """
    w = max(1, int(total * warmup))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on the convolution and linear weights only, not on biases and norms (lab 4).

    Tensors with more than one dimension are convolution kernels and linear
    weight matrices; the one-dimensional ones are biases and the BatchNorm
    scales and shifts. Pulling those towards zero would only fight the
    normalisation, so they get no decay.
    """
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def pixel_stats(ds: Photos, n: int = 200) -> tuple[float, float]:
    """Mean and standard deviation of the pixels of up to `n` training photographs.

    The photographs are spread evenly over the table, not the first `n`, which
    would all come from the same few photographers. The letterbox's grey border
    is included, as it is in every image the network sees.
    """
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    x = torch.stack([ds[i][0] for i in idx])
    return float(x.mean()), float(x.std())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the reference model")
    ap.add_argument("--fold", required=True, help="a fold number to validate on, or 'all'")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    ap.add_argument("--out", default="models/reference")
    a = ap.parse_args()

    # --- 1. Config and seed. Every number that changes the result comes from
    # the config; --seed exists only to measure how much the score moves
    # between seeds (a rerun with another seed is not an experiment step).
    cfg = load_config(a.config) if a.config else load_config()
    if a.seed is not None:
        cfg["seed"] = a.seed
    seed_everything(cfg["seed"])
    d, t = cfg["data"], cfg["train"]

    # --- 2. Folds. Every unit of `group_strict` lands whole in one fold, so no
    # page and no layout is on both sides of the split. The assignment is
    # deterministic: the same train.csv gives the same folds on every machine,
    # which is what makes fold scores comparable between your runs.
    df = train_table(cfg)
    df["fold"] = assign_folds(df, cfg["split"]["column"], cfg["split"]["n_folds"])
    if a.fold == "all":
        tr, va = df, df.iloc[:0]                    # everything trains, nothing validates
    else:
        k = int(a.fold)
        tr, va = df[df["fold"] != k], df[df["fold"] == k]
    # The run's name ends up in every file it writes; keep it unique if you
    # change something that is not in it (or pass another --out).
    name = f"smallcnn_{d['image_size']}_fold{a.fold}_seed{cfg['seed']}"
    print(f"{name}: {len(tr)} training photographs, {len(va)} validation")

    # --- 3. Data loaders. They read the 1024 px copies written by prepare.py
    # (run it first) and letterbox them to image_size grey squares. Only the
    # training loader augments and shuffles; it also drops the last, smaller
    # batch, so that every step's gradient is averaged over the same number of
    # photographs. The generator seeds the shuffling, worker_init the
    # augmentation in each worker process.
    def loader(rows, augment):
        ds = Photos([cache_path(cfg, "train", i) for i in rows["image_id"]], rows[TARGET],
                    size=d["image_size"], augment=augment, max_shift=t["max_shift"])
        g = torch.Generator().manual_seed(cfg["seed"])
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=augment, drop_last=augment,
                          num_workers=t["num_workers"], pin_memory=True, generator=g,
                          worker_init_fn=worker_init, persistent_workers=t["num_workers"] > 0)

    train_dl = loader(tr, augment=True)
    val_dl = loader(va, augment=False) if len(va) else None

    # --- 4. The network. `prior` sets the output bias so that the untrained
    # network already predicts the training mean. The input statistics are
    # measured on the training photographs of this run only (never on the
    # validation fold) and stored inside the model, so they travel with the
    # checkpoint and predict.py applies exactly the same standardisation.
    dev = device()
    model = build_model(cfg, prior=float(tr[TARGET].mean()))
    mean, std = pixel_stats(Photos([cache_path(cfg, "train", i) for i in tr["image_id"]],
                                   size=d["image_size"]))
    model.in_mean.fill_(mean)
    model.in_std.fill_(std)
    model.to(dev)

    # --- 5. Optimiser and schedule. AdamW with decoupled weight decay; the
    # scheduler is stepped after every batch, so `total` counts batches.
    opt = torch.optim.AdamW(param_groups(model, t["weight_decay"]), lr=t["lr"])
    total = t["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, t["warmup"]))

    log = []
    for epoch in range(1, t["epochs"] + 1):
        model.train()                               # BatchNorm uses batch statistics
        start, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            # The forward pass in bfloat16 where the GPU supports it (faster,
            # half the memory); the loss in full precision.
            with autocast(dev, t["amp"]):
                logit = model(x)
            # MSE on the sigmoid of the logit: the prediction stays in [0, 1]
            # like the label, and the loss is the square of the metric (RMSE).
            loss = torch.nn.functional.mse_loss(torch.sigmoid(logit.float()), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clipping the gradient norm at 1 stops a single bad batch from
            # making a huge step; in a normal step it does nothing.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        # train_rmse is measured on augmented photographs while the weights were
        # still changing, so it is only a rough guide; val_rmse (fold runs) is
        # computed after the epoch on every photograph of the held-out fold.
        row = {"epoch": epoch, "train_rmse": math.sqrt(np.mean(losses)),
               "seconds": round(time.time() - start)}
        if val_dl is not None:
            row["val_rmse"] = rmse(va[TARGET], predict(model, val_dl, t["amp"]))
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in row.items()), flush=True)

    # --- 6. Outputs. The model after the last epoch is kept, never the epoch
    # with the best validation score: picking the best epoch on the fold you
    # report would flatter the score, and the `all` run has no fold to pick on.
    # The config goes into the checkpoint, so predict.py rebuilds the same
    # network and image size from the checkpoint alone.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict(), "fold": a.fold},
               out / f"{name}.pt")
    pd.DataFrame(log).to_csv(out / f"{name}_log.csv", index=False)
    if val_dl is not None:
        # One row per validation photograph: label, prediction and the columns
        # you need for error analysis. Pool the four folds' files for the
        # out-of-fold RMSE in EXPERIMENTS.md.
        p = predict(model, val_dl, t["amp"])
        res = va[["image_id", TARGET, "eval_ok", "family"]].assign(pred=p)
        res.to_csv(out / f"{name}_val.csv", index=False)
        # The reported score leaves out the 47 photographs with eval_ok = 0,
        # uploaded heavily downscaled (fine for training, not for measuring;
        # lab 3). The per-epoch val_rmse above counts them all, so the two
        # differ slightly.
        ok = res["eval_ok"] == 1
        print(f"{name}: validation RMSE {rmse(res.loc[ok, TARGET], res.loc[ok, 'pred']):.4f} "
              f"on {int(ok.sum())} photographs (eval_ok)")
    print(f"-> {out / (name + '.pt')}")


if __name__ == "__main__":
    main()
