# Lab 6: borrowed eyes, a pretrained network in your competition repository

In lab 5 you copied the reference solution into your own repository,
changed one value of it and measured it on four folds. Today you make a
bigger change: you replace the reference's network with a **ResNet-18 that
has already learned to see on ImageNet's 1.3 million photographs** (lecture
6, part 5). The reference's network is the small CNN you built in lab 4 and
trained in lab 5. It learned everything from our 1841 photographs alone. You then decide how much of it to retrain. The result is one step
in your `EXPERIMENTS.md`: four folds, an out-of-fold RMSE and an upload.

**There are no files to copy this time.** Every change is described below,
file by file. You make the changes yourself in your own competition
repository, the one you copied out of `reference_solution/` in lab 5. Your
repository may already differ from the reference, and nothing here will
overwrite your work. Where this page shows code, type it or paste it. Where
it says **TASK**, it tells you exactly what to write, but not how.

| | what | where | for whom |
|---|---|---|---|
| 1 | imports and the ImageNet statistics | `src/model.py` | given |
| **TASK 1** | `split_backbone` and `PretrainedCNN` | `src/model.py` | you write it |
| 2 | `build_model` and `load_checkpoint` know the new network | `src/model.py` | given |
| 3 | a pretrained network brings its own input statistics; the run's name | `src/train.py`, `scripts/sanity.py` | given |
| 4 | the checks of today's tasks | `scripts/check_lab6.py` (new) | given |
| **TASK 2** | freezing, and a smaller learning rate for the backbone | `src/model.py`, `src/optim.py`, `src/train.py` | you write it |
| 5 | the config | `configs/resnet18.yaml` (new) | given |
| **TASK 3** | four folds, the out-of-fold RMSE, a log row, an upload | `EXPERIMENTS.md`, Kaggle | you run it |
| ⭐ | 768 px; another backbone; how much to freeze | | if you are ahead, or on a GPU |
| | Kaggle notebook with two GPUs | [the last section](#two-gpus-for-free-kaggle-notebooks) | if you have no GPU |

## How to work today

Every code block on this page is complete: copy it with the copy button at
its top right (on GitHub) and paste it where the text says. In the two TASKs you get a
**skeleton**. Paste it, then replace only the lines marked `# TODO`. Under
each TODO there is a folded hint (click ▸ to open it), and at the end of
each TASK the full solution is folded too. Use them in this order:

1. **Try the TODO yourself.** Look at `SmallCNN` in the same file, the shape
   table, and lecture 6.
2. **Stuck for more than 5 minutes?** Open that TODO's hint.
3. **At the clock time below, open the solution, paste it and read it.**
   Starting the four folds on time matters more today than writing every
   line yourself. Every question in the room is about *why* a line is
   there, and the hints say why.

| clock | where you should be | check |
|---|---|---|
| 0:10 | steps 1 to 4 pasted, TASK 1 started | |
| 0:35 | TASK 1 done (solution open at 0:30 at the latest) | `check_lab6.py`: every TASK 1 line PASS |
| 0:50 | TASK 2 done (solution open at 0:45 at the latest), step 5 pasted | `check_lab6.py`: all PASS |
| 0:55 | the four folds running (GPU, or Kaggle) | the first epoch's line appears |
| 1:30 | TASK 3: the row in `EXPERIMENTS.md`, the upload (or at home, when the folds are done) | your row on the leaderboard |

Already confident? Skip the skeletons: the shape table, the hints and the
check script are a complete specification.

## Before the lab

1. **Lab 5 is done.** Your repository trains the reference
   (`python -m src.train --fold 0`). `data/raw/` holds `train.csv`, `train/`,
   `test/` and `sample_submission.csv`. `python -m src.prepare` has run, and
   `scripts/oof.py` (lab 5, TASK 2) prints an out-of-fold RMSE.

**Lab 5 not finished?** You can still do today's lab. It needs only the
reference solution, unchanged, in a repository of your own, and one script
from lab 5. Open the block below.

<details><summary>Catching up on lab 5: your own repository and <code>scripts/oof.py</code></summary>

* Copy `reference_solution/` out of the course repository
  (<https://github.com/BME-SmartLab-Education/dl2026-course>) into a folder
  of its own, make it a git repository and push it to GitHub, as
  [the reference solution's README](../reference_solution/README.md)
  describes. That is your competition repository.
* Put the data into `data/raw/` and run `python -m src.prepare`, as in the
  same README.
* Save the script below as `scripts/oof.py` (lab 5's TASK 2, solved). It
  reads the four `_val.csv` files that the four fold runs write and prints
  one out-of-fold RMSE: the number that goes into `EXPERIMENTS.md`.

```python
"""The pooled out-of-fold RMSE of one experiment: lab 5, TASK 2, solved.

    python scripts/oof.py models/my_try

Reads the four validation files a fold run writes (*_fold<k>_*_val.csv), keeps
the eval_ok photographs, and prints the RMSE of each fold and of the four
pooled together. The pooled number is the one that goes into EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled out-of-fold RMSE of one experiment")
    ap.add_argument("folder", help="the --out folder of the four fold runs")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.folder}/*_fold[0-9]_*_val.csv"))
    if len(files) != 4:
        raise SystemExit(f"{len(files)} validation files in {a.folder}, expected 4: "
                         f"one experiment (one seed) per folder")
    d = pd.concat([pd.read_csv(f).assign(file=f) for f in files])
    d = d[d["eval_ok"] == 1]
    sq = (d["pred"] - d["usability"]) ** 2
    for f, e in sq.groupby(d["file"]):
        print(f"{f}: RMSE {np.sqrt(e.mean()):.4f} on {len(e)} photographs")
    print(f"out-of-fold RMSE {np.sqrt(sq.mean()):.4f} on {len(d)} photographs")


if __name__ == "__main__":
    main()
```

* Commit and push. Lab 5's measured step (one changed value, four folds)
  is missing from your `EXPERIMENTS.md`. Today's step counts on its own,
  and you can add lab 5's later.

</details>

2. **Commit.** `git status` must be empty, so that today's changes are one
   clean diff that you can look at (and undo).
3. **Download the ImageNet weights once** (45 MB, into `.cache/torch`, which
   git ignores):

   ```bash
   docker compose run --rm dev python -c "from torchvision import models; models.resnet18(weights='DEFAULT')"
   ```

4. **A GPU for TASK 3**: your own, or the Kaggle notebook of the
   [last section](#two-gpus-for-free-kaggle-notebooks), set up and tested
   before the lab. On a laptop CPU, today's code runs and its checks pass,
   but four folds of a ResNet-18 take hours (see [Compute](#compute)).

Nothing needs to be installed and the Docker image is not rebuilt:
`torchvision`, which holds the ImageNet networks, is already in it.

## The idea in one picture

```
grey photograph (1, 512, 512)
   │  repeat into 3 channels, standardise with ImageNet's mean and std
   ▼
trunk   = input block + layer1 + layer2 + layer3   ImageNet weights   (256, 32, 32)
   ▼
tail    = layer4                                  ImageNet weights   (512, 16, 16)
   ▼
[mean | max] over the 16 × 16 positions           as in lab 4         (1024,)
   ▼
dropout → Linear(1024, 1)                         new, ours          one logit
```

ResNet-18's own end (average pooling and a 1000-class layer) is cut off. In
its place goes lab 4's head: average **and** maximum pooling, then one
output. The trunk and the tail keep the weights that ImageNet taught them:
edges, strokes, textures, sharpness. That is what makes the difference.

---

## Step 1 (given): imports and the ImageNet statistics, `src/model.py`

Below the existing `import torch.nn as nn`, add:

```python
from torchvision import models

# lab 6: the statistics the ImageNet networks were trained with, per RGB channel
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

**Where these six numbers come from.** Take ImageNet's 1.3 million training
photographs and scale every pixel value from 0–255 to 0–1. Then, for the red
channel, put the red values of all pixels of all photographs into one long
list. Their average is 0.485, and their standard deviation is 0.229. Do the
same for green (0.456, 0.224) and for blue (0.406, 0.225).

The ResNet was trained on photographs standardised with these numbers,
`(pixel − mean) / std` per channel. So its first layer expects inputs that
are about 0 on average and spread about ±1. We feed our photographs the same
way, with ImageNet's numbers and not our own, because the network's weights
were learned for those. Our photographs are grey, so the grey value is
copied into all three channels before the standardisation (TODO 5).

## TASK 1: `split_backbone` and `PretrainedCNN`, `src/model.py`

**Paste the skeleton below into `src/model.py`, between `class SmallCNN` and
`def build_model`.** Your repository still imports with the skeleton in it,
but the network cannot be built until the TODOs are filled in. You change
only the lines marked `# TODO`. Everything else is given. Each TODO has a
hint below it (click to open), and at the end there is the full solution.

```python
def split_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    """A torchvision ImageNet network cut in two: the trunk, the last stage, and that stage's width."""
    net = getattr(models, arch)(weights="DEFAULT" if pretrained else None)   # None: random weights
    if arch.startswith("resnet"):
        trunk = nn.Sequential(net.conv1, ...)                 # TODO 1: the input block, then layer1, layer2, layer3
        return trunk, ..., ...                                # TODO 2: the tail (layer4) and its output width
    raise ValueError(f"unknown backbone {arch!r}")


class PretrainedCNN(nn.Module):
    """An ImageNet network with its classifier replaced by lab 4's head: one logit."""

    def __init__(self, arch: str = "resnet18", pretrained: bool = True, freeze: str = "none",
                 dropout: float = 0.0, prior: float = 0.5):
        super().__init__()
        self.trunk, self.tail, width = split_backbone(arch, pretrained)
        self.drop = nn.Dropout(dropout)
        self.head = ...                                       # TODO 3: [mean | max] of `width` channels in, 1 out
        # TODO 4: two buffers, rgb_mean and rgb_std, from IMAGENET_MEAN and IMAGENET_STD, shape (1, 3, 1, 1)

        # the head starts as SmallCNN's does: the untrained network predicts the training mean
        p = min(max(float(prior), 1e-3), 1 - 1e-3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(p / (1 - p)))
        self.freeze = freeze                                  # TASK 2 replaces this line

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W) grey photographs in [0, 1], as Photos returns them
        x = ...                                               # TODO 5: three channels, standardised like ImageNet
        f = self.tail(self.trunk(x))
        f = ...                                               # TODO 6: mean and max over the positions, side by side
        return self.head(self.drop(f)).squeeze(1)
```

<details><summary>Hint for TODO 1: what is in the trunk?</summary>

Print the network once and look at the names of its parts:
`docker compose run --rm dev python -c "from torchvision import models; print(models.resnet18())"`.
The input block (the first layers, which shrink the photograph 4×) is `conv1`, `bn1`, `relu`, `maxpool`, in this order. Then come
`layer1`, `layer2` and `layer3`. Leave out `layer4` (the tail), `avgpool`
and `fc`: our head replaces those two. Each one is an attribute: `net.layer1`.
</details>

<details><summary>Hint for TODO 2: the tail and its width</summary>

The tail is `net.layer4`. Its output has as many channels as the network's
own last layer `fc` has inputs: `net.fc.in_features` (512 for ResNet-18,
2048 for ResNet-50). Return the three things as a tuple: `trunk, tail, width`.
</details>

<details><summary>Hint for TODO 3: how wide is the head?</summary>

`forward` puts the mean and the maximum of each of the `width` channels side
by side: `2 * width` numbers go in, one comes out. It is `nn.Linear`, as in `SmallCNN`.
</details>

<details><summary>Hint for TODO 4: buffers</summary>

`self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))`,
and the same for `rgb_std`. The shape `(1, 3, 1, 1)` broadcasts against
`(N, 3, H, W)`: one number per channel. A buffer, not a plain attribute, so
that `.to(device)` moves it to the GPU and the checkpoint saves it. Unlike
a parameter, nothing trains it.
</details>

<details><summary>Hint for TODO 5: three channels, standardised</summary>

`x.expand(-1, 3, -1, -1)` repeats the one grey channel three times without
copying memory (`-1` means "keep this size"). Then subtract `self.rgb_mean`
and divide by `self.rgb_std`. One line: `x = (x.expand(...) - ...) / ...`.

Why ImageNet's statistics and not ours: the trunk learned its filters on
photographs standardised this way, so it expects inputs on that scale. This
is also why `PretrainedCNN` has no `in_mean` and `in_std`, and why step 3
skips measuring them.
</details>

<details><summary>Hint for TODO 6: mean and max</summary>

`f` is `(N, width, h, w)`. Look at `SmallCNN.forward` a few lines above:
`torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)`, which gives `(N, 2 * width)`.
</details>

What you get for ResNet-18 at 512 px. The check script of step 4 compares
your network with this table:

| part | output shape | parameters |
|---|---|---|
| input, repeated and standardised | (N, 3, 512, 512) | 0 |
| trunk: conv1 7×7/2, bn1, relu, maxpool 3×3/2 | (N, 64, 128, 128) | 9,536 |
| trunk: layer1 | (N, 64, 128, 128) | 147,968 |
| trunk: layer2 | (N, 128, 64, 64) | 525,568 |
| trunk: layer3 | (N, 256, 32, 32) | 2,099,712 |
| tail: layer4 | (N, 512, 16, 16) | 8,393,728 |
| mean and max pooling, concatenated | (N, 1024) | 0 |
| dropout, Linear(1024, 1), squeeze | (N,) | 1,025 |
| **total** | | **11,177,537** (trunk 2,782,784) |

<details><summary>Stuck, or behind the clock at 0:30? The full solution of TASK 1</summary>

Replace your skeleton with this. Then read it line by line against the hints:
the questions in the room are about why each line is there.

```python
def split_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    """A torchvision ImageNet network cut in two: the trunk, the last stage, and that stage's width."""
    net = getattr(models, arch)(weights="DEFAULT" if pretrained else None)   # None: random weights
    if arch.startswith("resnet"):
        trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                              net.layer1, net.layer2, net.layer3)
        return trunk, net.layer4, net.fc.in_features
    raise ValueError(f"unknown backbone {arch!r}")


class PretrainedCNN(nn.Module):
    """An ImageNet network with its classifier replaced by lab 4's head: one logit."""

    def __init__(self, arch: str = "resnet18", pretrained: bool = True, freeze: str = "none",
                 dropout: float = 0.0, prior: float = 0.5):
        super().__init__()
        self.trunk, self.tail, width = split_backbone(arch, pretrained)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * width, 1)
        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # the head starts as SmallCNN's does: the untrained network predicts the training mean
        p = min(max(float(prior), 1e-3), 1 - 1e-3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(p / (1 - p)))
        self.freeze = freeze                                  # TASK 2 replaces this line

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W) grey photographs in [0, 1], as Photos returns them
        x = (x.expand(-1, 3, -1, -1) - self.rgb_mean) / self.rgb_std
        f = self.tail(self.trunk(x))
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        return self.head(self.drop(f)).squeeze(1)
```
</details>

## Step 2 (given): `build_model` and `load_checkpoint`, `src/model.py`

Replace `build_model` with:

```python
def build_model(cfg: dict, prior: float = 0.5, pretrained: bool = True) -> nn.Module:
    """The network the config names. `pretrained=False` only skips the ImageNet download
    (load_checkpoint overwrites every weight anyway)."""
    m = cfg["model"]
    name = m.get("name", "smallcnn")
    if name == "smallcnn":
        return SmallCNN(m["widths"], m["stem_stride"], m["stem_kernel"], m["dropout"], prior)
    return PretrainedCNN(name, pretrained=pretrained and m.get("pretrained", True),
                         freeze=m.get("freeze", "none"), dropout=m.get("dropout", 0.0), prior=prior)
```

and in `load_checkpoint`, change `model = build_model(cfg)` to

```python
    model = build_model(cfg, pretrained=False)
```

A config without `model.name` is the reference. Your old configs and
checkpoints therefore keep working, and `predict.py` needs no change: it
rebuilds whatever network the checkpoint's config names.

## Step 3 (given): `src/train.py` and `scripts/sanity.py`

**`src/train.py`, the run's name.** The line that starts with
`name = f"smallcnn_` becomes

```python
    name = f"{cfg['model'].get('name', 'smallcnn')}_{d['image_size']}_fold{a.fold}_seed{cfg['seed']}"
```

**`src/train.py`, step 4 of `main`.** Only `SmallCNN` measures its input
statistics. Replace the four lines from `mean, std = pixel_stats(` to
`model.in_std.fill_(std)` with

```python
    if hasattr(model, "in_mean"):                   # a pretrained network brings its own statistics
        mean, std = pixel_stats(Photos([cache_path(cfg, "train", i) for i in tr["image_id"]],
                                       size=d["image_size"]))
        model.in_mean.fill_(mean)
        model.in_std.fill_(std)
```

**`scripts/sanity.py`, in `fresh`.** The same idea: the two `fill_` lines become

```python
    if hasattr(model, "in_mean"):                   # a pretrained network brings its own statistics
        model.in_mean.fill_(float(x.mean()))
        model.in_std.fill_(float(x.std()))
```

## Step 4 (given): the checks, `scripts/check_lab6.py`

Create the file with this content. It checks TASK 1 against the table above,
and TASK 2 against the next section.

<details><summary>scripts/check_lab6.py: click to open, then copy all of it</summary>

```python
"""Lab 6: the checks of TASK 1 (PretrainedCNN) and TASK 2 (freezing, the backbone's learning rate).

    python scripts/check_lab6.py            # after TASK 1: the TASK 1 lines pass, the TASK 2 lines fail
    python scripts/check_lab6.py            # after TASK 2: everything passes

Runs on a CPU in under a minute. The first run downloads the ResNet-18
weights (45 MB) into .cache/torch.
"""
from __future__ import annotations

import sys

import torch
from torchvision import models

from src.config import load_config
from src.model import PretrainedCNN, build_model
from src.optim import param_groups

TRUNK, TAIL, HEAD = 2_782_784, 8_393_728, 1_025       # ResNet-18: input block + layer1-3, layer4, Linear(1024, 1)
failed = []


def check(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}{': ' + detail if detail else ''}")
    if not ok:
        failed.append(what)


def n_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


print("TASK 1: PretrainedCNN")
torch.manual_seed(0)
try:
    model = PretrainedCNN("resnet18", pretrained=True, prior=0.8).eval()
    with torch.no_grad():
        model(torch.rand(1, 1, 64, 64))
except Exception as e:                                  # a TODO still open, or a typo
    print(f"  FAIL  the network cannot be built or run yet: {type(e).__name__}: {e}")
    print("        Fill in the TODOs of TASK 1 (hints under each one), then run this again.")
    sys.exit(1)
counts = (n_params(model.trunk), n_params(model.tail), n_params(model.head))
check(counts == (TRUNK, TAIL, HEAD), "parameters of trunk, tail, head",
      f"{counts}, expected {(TRUNK, TAIL, HEAD)}")
ref = models.resnet18(weights="DEFAULT")
check(torch.equal(model.trunk[0].weight, ref.conv1.weight) and
      torch.equal(model.tail[1].conv2.weight, ref.layer4[1].conv2.weight),
      "the ImageNet weights are loaded (conv1 and layer4 equal torchvision's)")
x = torch.rand(3, 1, 512, 512)                          # three grey photographs, as Photos returns them
with torch.no_grad():
    maps = model.trunk(x.expand(-1, 3, -1, -1))
    out = model(x)
check(tuple(maps.shape) == (3, 256, 32, 32), "trunk output at 512 px", f"{tuple(maps.shape)}, expected (3, 256, 32, 32)")
check(tuple(out.shape) == (3,), "one logit per photograph", f"output shape {tuple(out.shape)}")
p = torch.sigmoid(out)
check(bool(((p - 0.8).abs() < 0.02).all()), "the untrained network predicts the prior",
      f"{p.min():.3f} to {p.max():.3f}, expected about 0.8")
check(not hasattr(model, "in_mean"), "no in_mean: the ImageNet statistics are fixed buffers")
cfg = load_config()
cfg["model"] = {"name": "resnet18", "pretrained": True, "freeze": "none", "dropout": 0.0}
check(isinstance(build_model(cfg, prior=0.8, pretrained=False), PretrainedCNN),
      "build_model returns a PretrainedCNN for model.name resnet18")

print("TASK 2: freezing and the backbone's learning rate")
try:
    for freeze, expect in (("none", TRUNK + TAIL + HEAD), ("trunk", TAIL + HEAD), ("backbone", HEAD)):
        m = PretrainedCNN("resnet18", pretrained=False, freeze=freeze).train()
        bn_training = [mod.training for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
        n_frozen_bn = {"none": 0, "trunk": 15, "backbone": 20}[freeze]   # ResNet-18 has 20 BatchNorm layers
        check(n_params(m, trainable_only=True) == expect, f"freeze={freeze}: trainable parameters",
              f"{n_params(m, trainable_only=True):,}, expected {expect:,}")
        check(bn_training.count(False) == n_frozen_bn, f"freeze={freeze}: BatchNorm layers in eval mode after train()",
              f"{bn_training.count(False)}, expected {n_frozen_bn}")
    m = PretrainedCNN("resnet18", pretrained=False, freeze="trunk")
    groups = param_groups(m, 0.01, 1e-3, 0.1)
    in_groups = sum(p.numel() for g in groups for p in g["params"])
    lrs = sorted({g["lr"] for g in groups})
    check(in_groups == TAIL + HEAD, "param_groups leaves the frozen parameters out", f"{in_groups:,}")
    check(lrs == [1e-4, 1e-3], "learning rates: backbone lr x backbone_lr, head lr", f"{lrs}")
    head_lr = {g["lr"] for g in groups if any(p is m.head.weight for p in g["params"])}
    check(head_lr == {1e-3}, "the head learns at lr", f"{head_lr}")
except (KeyError, TypeError) as e:
    print(f"  not yet: TASK 2 is not done ({type(e).__name__}: {e})")
    failed.append("TASK 2")

print("all lab 6 checks passed" if not failed else f"{len(failed)} failed: {', '.join(failed)}")
sys.exit(1 if failed else 0)
```
</details>

Run it:

```bash
docker compose run --rm dev python scripts/check_lab6.py
```

After TASK 1, every line under "TASK 1" must say PASS. The lines under
"TASK 2" fail until you have done TASK 2.

## TASK 2: how much to retrain, and how fast

Lecture 6 named three choices: keep the pretrained network **frozen** and
train only a new head; **fine-tune** all of it; or **train from scratch**.
There is a middle ground, too: freeze the trunk and retrain only the last
stage. TASK 2 makes each of them one line of the config.

### TASK 2a: freezing, `src/model.py`

In `PretrainedCNN.__init__`, replace the line `self.freeze = freeze` with
the first block below. Paste the `train` method into the class, between
`__init__` and `forward`:

```python
        # TASK 2a: the parts that do not learn
        self.frozen = {"none": [], "trunk": ..., "backbone": ...}[freeze]   # TODO 7: lists of parts
        for part in self.frozen:
            ...                                                              # TODO 8: no gradient for this part
```

```python
    def train(self, mode: bool = True) -> "PretrainedCNN":
        super().train(mode)
        ...                                                                  # TODO 9: frozen parts back to eval mode
        return self
```

Why `train` has to be overridden: `model.train()`, called by `train.py` at
every epoch, switches every BatchNorm layer to training mode. In training
mode, BatchNorm normalises with the batch's statistics and **updates its
running mean and variance**, with or without gradients. Without the
override, a "frozen" trunk would drift away from ImageNet anyway, one batch
of our pages at a time.

<details><summary>Hint for TODO 7</summary>

`"trunk"` freezes the trunk: `[self.trunk]`. `"backbone"` freezes the trunk
and the tail, so only the head learns: `[self.trunk, self.tail]`. A plain
Python list, not an `nn.ModuleList`: the parts are already registered as
`self.trunk` and `self.tail`. Any other value of `freeze` raises a
`KeyError`, which is what we want for a typo in a config.
</details>

<details><summary>Hint for TODO 8</summary>

`part.requires_grad_(False)` switches off the gradient of every parameter
in the part. Autograd then builds no graph through it, and the optimiser
never changes it.
</details>

<details><summary>Hint for TODO 9</summary>

A loop over `self.frozen` that calls `part.eval()` on each. `eval()` is
recursive: every BatchNorm inside the part goes back to its stored
(ImageNet) statistics.
</details>

### TASK 2b: a smaller learning rate for the pretrained part, `src/optim.py`

The new head starts from random weights. The backbone starts from good
ones, and large steps would destroy them. So the head learns at `lr` and
everything else at `lr * backbone_lr`. **Replace `param_groups` in
`src/optim.py` with:**

```python
def param_groups(model: torch.nn.Module, weight_decay: float, lr: float,
                 backbone_lr: float = 1.0) -> list[dict]:
    """Weight decay on weights only, not on biases and norms (lab 4); the head learns at lr,
    the rest (the pretrained backbone) at lr * backbone_lr (lab 6). Frozen parameters are left out."""
    groups = []
    for is_head in (False, True):
        ps = [p for n, p in model.named_parameters()
              if ...]                                    # TODO 10: trainable, and in the head exactly when is_head
        rate = ...                                       # TODO 11: lr for the head, lr * backbone_lr for the rest
        groups += [{"params": [p for p in ps if p.ndim > 1], "weight_decay": weight_decay, "lr": rate},
                   {"params": [p for p in ps if p.ndim <= 1], "weight_decay": 0.0, "lr": rate}]
    return [g for g in groups if g["params"]]            # with freeze: backbone only the head is left
```

<details><summary>Hint for TODO 10</summary>

Two conditions, joined with `and`: `p.requires_grad` (a frozen parameter has
nothing to do in the optimiser), and `n.startswith("head.") == is_head`. The
head's parameters are named `head.weight` and `head.bias`. `SmallCNN`'s last
layer is called `head` too, so the reference keeps working: with
`backbone_lr` 1, every group gets `lr`, as before.
</details>

<details><summary>Hint for TODO 11</summary>

`rate = lr if is_head else lr * backbone_lr`. The warm-up-cosine schedule
multiplies each group's own rate, so the ratio holds for the whole run.
</details>

**Then, in `src/train.py`, the optimiser line** (given) becomes

```python
    opt = torch.optim.AdamW(param_groups(model, t["weight_decay"], t["lr"], t.get("backbone_lr", 1.0)),
                            lr=t["lr"])
```

Run `scripts/check_lab6.py` again. Every line must say PASS.

<details><summary>Stuck, or behind the clock at 0:45? The full solution of TASK 2</summary>

In `PretrainedCNN.__init__`, instead of `self.freeze = freeze`:

```python
        # TASK 2a: the parts that do not learn
        self.frozen = {"none": [], "trunk": [self.trunk],
                       "backbone": [self.trunk, self.tail]}[freeze]
        for part in self.frozen:
            part.requires_grad_(False)
```

The method, in `PretrainedCNN`:

```python
    def train(self, mode: bool = True) -> "PretrainedCNN":
        super().train(mode)
        for part in self.frozen:          # frozen BatchNorm keeps ImageNet's statistics
            part.eval()
        return self
```

The two TODO lines of `param_groups`:

```python
        ps = [p for n, p in model.named_parameters()
              if p.requires_grad and n.startswith("head.") == is_head]
        rate = lr if is_head else lr * backbone_lr
```
</details>

## Step 5 (given): the config, `configs/resnet18.yaml`

Create `configs/resnet18.yaml` with this content. It is `reference.yaml`
with the `model:` section replaced and five values changed (each one says
what the reference had):

```yaml
# Lab 6: the reference's pipeline with an ImageNet ResNet-18 (src/model.py, PretrainedCNN).
# python -m src.train --fold 0 --config configs/resnet18.yaml --out models/resnet18

seed: 42

data:
  raw: data/raw                  # train.csv, train/, test/, sample_submission.csv
  work: data/work/reference      # resized copies, written by prepare.py (the same as the reference's)
  cache_size: 1024               # long side of the resized copies
  image_size: 512                # lab 6: 512 px (the reference: 768)

split:
  column: group_strict           # never split at random: see the README
  n_folds: 4                     # the 6 strict units of the training set in 4 folds

model:
  name: resnet18                 # smallcnn (the reference) | resnet18 | resnet34 | resnet50
  pretrained: true               # false: the same network from random weights
  freeze: none                   # none | trunk (only layer4 and the head learn) | backbone (only the head)
  dropout: 0.0

train:
  epochs: 12
  batch_size: 16
  lr: 3.0e-4                     # the head's learning rate (the reference: 2.0e-3)
  backbone_lr: 1.0               # the backbone learns at lr x backbone_lr
  weight_decay: 0.01             # the reference: 0.05
  warmup: 0.1                    # fraction of the steps with a linearly rising learning rate
  max_shift: 16                  # augmentation: mirror and shift only, never blur or brightness
  amp: true                      # bfloat16 on a GPU that supports it; false on Kaggle's T4
  num_workers: 4

wandb:
  project: snap-or-retake        # where train.py logs; WANDB_MODE in .env turns it offline or off
```

Why these numbers:
* **lr 0.0003, not 0.002.** Pretrained weights need much smaller steps than
  random ones, or the first updates destroy what ImageNet taught them.
* **Weight decay 0.01, not 0.05.** It pulls the weights towards zero, and
  zero is not where pretrained weights should end up.
* **512 px, not 768.** A ResNet-18 at 768 px needs about 2.3 times the
  computation, which matters on Kaggle's T4. 768 px is extra E1.
* **A batch of 16** at 512 px fits a laptop GPU.

Check the config with lab 4's checks (about a minute on a GPU; on a CPU add
`--steps 30` and accept a loss that is still falling):

```bash
docker compose run --rm dev python scripts/sanity.py --config configs/resnet18.yaml
```

## TASK 3: the step, measured and uploaded

**1. Four folds** (from your repository root; bash and PowerShell alike):

```bash
docker compose run --rm dev bash -c 'for k in 0 1 2 3; do \
    python -m src.train --fold $k --config configs/resnet18.yaml --out models/resnet18; done'
```

**2. The out-of-fold RMSE** with your lab-5 script:

```bash
docker compose run --rm dev python scripts/oof.py models/resnet18
```

**3. One row in `EXPERIMENTS.md`**: the change ("pretrained ResNet-18,
512 px"), why you expected it to help, the config, the four fold RMSEs and
the pooled one. It is a step if it beats the row it builds on.

**4. Predict with the four fold models, check, upload** (the same commands
as in lab 5):

```bash
docker compose run --rm dev bash -c 'python -m src.predict models/resnet18/*_fold?_*.pt --out submissions/resnet18.csv'
docker compose run --rm dev python scripts/check_submission.py submissions/resnet18.csv
```

Upload `submissions/resnet18.csv` on Kaggle (*Submit Prediction*) with a
description that names the config and its out-of-fold RMSE.

**5. Commit**: the code, `configs/resnet18.yaml`, `EXPERIMENTS.md`. Leave
out `models/` and `submissions/`; `.gitignore` already does that.

What to expect, from the trial: `configs/resnet18.yaml` gave **0.148** out of
fold (folds 0.142, 0.130, 0.167, 0.142) against the reference's 0.194 (0.169,
0.184, 0.230, 0.178), and a public score of about 0.17 against the
reference's 0.216. Every fold is better: a clear step. Yours will differ by
a few thousandths (another seed, another GPU), more on Kaggle without mixed
precision.

## ⭐ Extras: if you are ahead, or on a GPU

Each extra is one more config and four more folds. Each is a candidate step
for your log, if it beats the step before it.

**⭐ E1: 768 px.** Copy `configs/resnet18.yaml` to
`configs/resnet18_768.yaml` and set `image_size: 768`. The trial: 0.139 out
of fold against 0.148 at 512 px. Our labels are about fine blur, and fine
blur needs pixels.

**⭐ E2: how much to retrain.** Three copies of `resnet18.yaml`, one value
each: `freeze: backbone` with `lr: 1.0e-3`; `freeze: trunk`; and
`pretrained: false`. The result, against the fine-tuned network, is the
lecture's question measured on our task. The trial, against 0.148 for the
fine-tuned network: frozen backbone 0.191, frozen trunk 0.145, random weights
0.205. A fourth copy puts TASK 2b to work: `lr: 1.0e-3` (the new head) with
`backbone_lr: 0.3` (the backbone at 0.0003) gave 0.141.

**⭐ E3: another backbone.** Extend `split_backbone` with the
EfficientNet and ConvNeXt families of torchvision:

```python
    if arch.startswith(("efficientnet", "convnext")):
        return net.features[:6], net.features[6:], net.classifier[-1].in_features
```

In both families, `features` is a list of stages, and the last stride-32
stages are the tail. Then `name: efficientnet_b0` with `lr: 1.0e-3`, or
`name: convnext_tiny` with `lr: 3.0e-4`. Expect 2 to 4 times the time of
ResNet-18. The first run downloads the weights (21 and 110 MB). `freeze: trunk`
works for both. The check script knows only ResNet-18. The trial at 768 px:
EfficientNet-B0 0.135 and ConvNeXt-tiny 0.121 out of fold. The four folds take
about 45 and 80 minutes on a laptop GPU.

## Compute

Measured on a laptop GPU (RTX 3500 Ada) and on four CPU cores
(`docker run --cpus 4`), for the four folds:

| run | laptop GPU | 4 CPU cores |
|---|---|---|
| the checks: `check_lab6.py`; `sanity.py` (`--steps 30` on a CPU) | seconds | under a minute; a few minutes |
| `configs/resnet18.yaml`, 512 px, 12 epochs | 10 to 15 min | over 12 hours: not an option |
| the same at 768 px (⭐ E1) | 15 to 20 min | |
| the CPU route below | 3 min | about an hour |

On a GPU, the time goes mostly into decoding the photographs, not into the
network, so 768 px costs less extra than the arithmetic suggests. Kaggle's T4
without mixed precision is slower than a laptop GPU. Expect 20 to 40 minutes
for the four folds at 512 px (not measured from here).

**The CPU route**, if you have no GPU today: the same network, a smaller
step. Freeze the trunk, use smaller photographs and fewer epochs. In a copy
`configs/resnet18_cpu.yaml`: `image_size: 384`, `freeze: trunk`,
`epochs: 3`. One epoch takes about 4 minutes on four cores, plus the
validation, so the four folds take about an hour (less with more cores).
Start it at the end of the lab and let it finish at home. It scored 0.169
out of fold in the trial, against the reference's 0.194, so it is a kept
step too. It is a different step from `resnet18.yaml`: log it under its own
config.

If a GPU runs out of memory: halve `batch_size` and say so in your log (the
learning rate is tuned for 16). Next, try `image_size: 512`, then a smaller
backbone.

## Two GPUs for free: Kaggle notebooks

Kaggle has a Jupyter notebook editor in the browser, attached to the
competition. It comes with two free NVIDIA T4 GPUs for a weekly quota: about 30
hours at the time of writing, shown in the notebook's *Session options*. Your
repository runs there unchanged, **without Docker**. Kaggle's notebook image
already has PyTorch, torchvision, OpenCV, pandas and W&B.

What changes against your laptop:

| | your laptop | a Kaggle notebook |
|---|---|---|
| a command | `docker compose run --rm dev python -m src.train ...` | `!python -m src.train ...` in a cell |
| your code | your repository | cloned from GitHub into `/kaggle/working/repo` |
| the photographs | `data/raw/` | read-only under `/kaggle/input/`, linked into `data/raw/` |
| what is kept | everything | only `/kaggle/working`, and only in a saved version (or with persistence on) |
| mixed precision | `amp: true` | `amp: false`: the T4 has no bfloat16 units |

**The photographs stay private.** Keep the notebook *Private* (the default).
Never make it public, and never put the photographs into a public dataset.

### Once: the account

1. **Phone verification.** On kaggle.com: your profile picture → *Settings*
   → *Phone verification*. Without it a notebook gets neither a GPU nor
   internet access.
2. **A GitHub token, if your repository is private** (skip this step for a
   public one), so that the notebook can clone it. On github.com: *Settings → Developer settings → Personal
   access tokens → Fine-grained tokens → Generate new token*. Under
   *Repository access* choose *Only select repositories* and pick your
   competition repository. Under *Permissions → Repository permissions →
   Contents* choose *Read-only*. Copy the token (`github_pat_...`); GitHub
   shows it only once.
3. **Your W&B key**, from <https://wandb.ai/authorize>.
4. **Push today's work** from your laptop (`git push`). The notebook sees
   only what is on GitHub.

### Once per notebook: create it, attach the data, switch on the GPU

5. **Create the notebook from the competition.** On the competition's page,
   open the *Code* tab and click *New Notebook*. The notebook opens with the
   competition's data already attached: the test photographs and
   `sample_submission.csv`, under `/kaggle/input/<competition name>/`.
   Rename it at the top left (e.g. `snap-or-retake-train`).
6. **Attach the training set.** The training photographs are not in the
   competition. They are the private dataset you downloaded in lab 3 (its
   link is in the competition's *Data* tab). In the notebook's right-hand
   panel, *Input → + Add Input*, then paste the dataset's link into the
   search box (or find it under *Your Work*/*Datasets* once you have opened
   it), and click **+**. It appears under *Input* as a second entry.
   If it does not show up, open the dataset's page and use *Code → New
   Notebook* there instead. That notebook has the training set attached,
   and you add the competition with *+ Add Input → Competitions*.
7. **Switch on the GPUs: choose `GPU T4 x2`.** In the right-hand panel,
   *Session options → Accelerator → GPU T4 x2* (or the *Settings →
   Accelerator* menu at the top). Confirm the restart. Cell 4 trains two
   folds at a time, one on each GPU. **Not P100**: Kaggle's current PyTorch may no longer
   support it. **Not TPU**: our code does not run on one. The quota counter
   next to the accelerator shows the hours used this week.
8. **Internet on**: *Session options → Internet → On*. It is needed for
   `git clone`, the ImageNet weights and W&B. The switch is greyed out
   until your phone is verified.
9. **Persistence** (optional): *Session options → Persistence → Files only*.
   Then `/kaggle/working` (your clone, the resized photographs, the
   checkpoints) survives a restart of the session, so cell 2 does not
   resize everything again.
10. **The secrets**: the menu *Add-ons → Secrets → + Add Secret*. Add
    `GITHUB_TOKEN` and `WANDB_API_KEY` with the values from steps 2 and 3,
    and tick both for this notebook. Secrets never appear in the notebook's
    text, so the token goes here and never into a cell.

### The notebook: four cells

**Cell 1, is the GPU there?** With *GPU T4 ×2* it prints two `Tesla T4`
lines and `True`. If it prints `False`, or a P100, go back to step 7.

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())
```

**Cell 2, your repository and the keys.** The notebook runs **your own
repository**: the copy of `reference_solution` you made in lab 5 (from the
course repository, <https://github.com/BME-SmartLab-Education/dl2026-course>),
with today's TASKs pushed in step 4. The course repository itself is not
enough: its `reference_solution` is lab 5's code, without today's
`PretrainedCNN`, `check_lab6.py` and `resnet18.yaml`.

Change only the two marked lines. `REPO` is the owner and the name in your
repository's address, `https://github.com/<owner>/<name>`, without
`https://` and `.git`. `CODE` stays `""` when `src/` is at the top of your
repository; if you keep the code in a folder, put its name there.

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

REPO = "your-github-name/your-repository"         # TODO: owner/name of YOUR repository on github.com
CODE = ""                                         # TODO only if src/ is in a folder of your repository
# The course repository, the one your copy comes from, would be:
#   REPO = "BME-SmartLab-Education/dl2026-course"     # owner/name on github.com
#   CODE = "reference_solution"                       # the folder in REPO that holds src/; "" if at the top
# It does NOT work as it is: its reference_solution is lab 5's code. Make the
# modifications described above (Steps 1-5, TASKs 1-2) in your own copy and
# push them first; the check at the end of this cell lists what is missing.
assert "your-github-name" not in REPO, "set REPO to your repository first"

secrets = UserSecretsClient()
def secret(name):
    try:
        return secrets.get_secret(name)
    except Exception:
        print(f"no secret {name}: is it added and ticked for this notebook (step 10)?")
        return ""

token = secret("GITHUB_TOKEN")                    # empty: works only for a public repository
wandb_key = secret("WANDB_API_KEY")
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
else:
    os.environ["WANDB_MODE"] = "disabled"         # trains without W&B
work = os.path.join("/kaggle/working/repo", CODE)
os.environ["PYTHONPATH"] = work                            # what the Docker image sets on your laptop
os.environ["TORCH_HOME"] = "/kaggle/working/.cache/torch"  # the ImageNet weights land here
os.environ["PYTHONUNBUFFERED"] = "1"                       # print each epoch as it ends

url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token or "\0", "***"))
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed: is REPO right ({REPO}), and may the token read it (step 2)?")

if not os.path.exists("/kaggle/working/repo/.git"):
    git("clone", url, "/kaggle/working/repo")
    git("-C", "/kaggle/working/repo", "remote", "set-url", "origin", f"https://github.com/{REPO}.git")
os.chdir("/kaggle/working/repo")
git("pull", "--ff-only", url)
git("log", "--oneline", "-1")
os.chdir(work)

# is today's work in it?
need = {"scripts/check_lab6.py": "Step 4", "configs/resnet18.yaml": "Step 5",
        "scripts/oof.py": "lab 5, TASK 2"}
missing = [f"{f} ({step})" for f, step in need.items() if not os.path.exists(f)]
if not os.path.exists("src/model.py") or "class PretrainedCNN" not in open("src/model.py").read():
    missing.append("class PretrainedCNN in src/model.py (TASK 1)")
if missing:
    raise RuntimeError(f"{REPO} has no lab 6 code yet. Missing: {', '.join(missing)}. "
                       "Do those steps on your laptop, commit, git push, then run this cell again.")
print("ready in", os.getcwd())
```

The `git log` line must show your latest commit, the one with today's TASKs,
and the last line `ready in /kaggle/working/repo`. If git fails, or today's
files are not in the repository, the cell stops and says why. The token is
printed as `***`.
The `set-url` line takes the token out of the clone's settings, so it is
never saved with the notebook's files. Running the cell again only pulls
your newest commits.

**Cell 3, the data.** The two inputs are mounted read-only under
`/kaggle/input/`. Their folder names depend on how they were attached, so
the cell finds them, links them where your code expects them, and resizes
the photographs once (`src.prepare`, about 3 minutes):

```python
import glob, os

def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, f"{name} not found under /kaggle/input: is that input attached (steps 5 and 6)?"
    return hits[0]

train_csv, sample = find("train.csv"), find("sample_submission.csv")
links = {"train.csv": train_csv, "train": os.path.join(os.path.dirname(train_csv), "train"),
         "sample_submission.csv": sample, "test": os.path.join(os.path.dirname(sample), "test")}
os.makedirs("data/raw", exist_ok=True)
for name, src in links.items():
    assert os.path.exists(src), src
    if not os.path.lexists(f"data/raw/{name}"):
        os.symlink(src, f"data/raw/{name}")
print({k: len(os.listdir(v)) if os.path.isdir(v) else v for k, v in links.items()})
!python -m src.prepare
```

It must print 1841 training and 1628 test photographs. To look at what is
attached yourself: `!ls /kaggle/input/*`.

**Cell 4, the work**: the same commands as on your laptop, without
`docker compose run --rm dev`, and with both GPUs busy.

* `%%bash` runs the cell as one shell script. `set -e` stops it at the
  first command that fails, so an error shows up once, at its cause,
  instead of being followed by a chain of others.
* The `sed` line writes a copy of the config for Kaggle,
  `configs/resnet18_kaggle.yaml`: mixed precision off, and 2 data-loading
  workers per run instead of 4, because two runs share the notebook's few
  CPU cores. Your `resnet18.yaml` stays as it is, so cell 2 can still pull
  your newest commits.
* **The two GPUs train two folds at the same time**: folds 0 and 1 first,
  then 2 and 3. `CUDA_VISIBLE_DEVICES=0` or `=1` shows one run only one of
  the GPUs, which it then sees as its own `cuda`. Each line of the output
  starts with its fold, `[fold 1] ...`, because the two runs print in
  turns.

```python
%%bash
set -euo pipefail
[ "$(nvidia-smi -L | wc -l)" -ge 2 ] || { echo "cell 4 needs two GPUs: Accelerator GPU T4 x2 (step 7)"; exit 1; }
sed -e 's/^  amp: true/  amp: false/' -e 's/^  num_workers: 4/  num_workers: 2/' \
    configs/resnet18.yaml > configs/resnet18_kaggle.yaml
python scripts/check_lab6.py

fold() {   # fold $1 on GPU $2, each output line marked with its fold
    CUDA_VISIBLE_DEVICES=$2 python -m src.train --fold $1 --config configs/resnet18_kaggle.yaml \
        --out models/resnet18 2>&1 | sed -u "s/^/[fold $1] /"
}
pair() {   # two folds at the same time, one per GPU; fails if either fails
    fold $1 0 & local a=$!
    fold $2 1 & local b=$!
    local ok=0; wait $a || ok=1; wait $b || ok=1; return $ok
}
pair 0 1
pair 2 3

python scripts/oof.py models/resnet18
python -m src.predict models/resnet18/*_fold?_*.pt --out submissions/resnet18.csv
python scripts/check_submission.py submissions/resnet18.csv
mkdir -p /kaggle/working/out && cp models/resnet18/*.csv submissions/resnet18.csv /kaggle/working/out/
```

`%%bash` must be the very first line of the cell. If `check_lab6.py`
reports a FAIL, fix that TASK on your laptop, push, and run cells 2 and 4
again. If one fold fails, the cell stops after its pair; the error is in
the lines marked with that fold.

While it runs, the GPU and CPU meters at the top right of the editor show
two busy GPUs. One epoch prints one line per fold, so a silent cell for
many minutes is normal during the first epoch.

### Running it: interactive, or in the background

* **Interactively** (the ▶ buttons, or *Run All*): good for cells 1 to 3
  and for trying things. The session ends when you close the tab or stay
  idle for too long, and without persistence its files go with it.
* **In the background, for the four folds**: *Save Version* (top right) →
  *Save & Run All (Commit)* → *Save*. Kaggle runs the whole notebook from
  the top on a fresh machine, with the same accelerator, up to 12 hours,
  and you can close the tab. The run appears under the notebook's
  *Versions*. When it is done, open that version: its *Output* tab holds
  `/kaggle/working`. Download `out/` (the four `_val.csv` files, the logs and
  the submission) and, if you want them, the checkpoints in
  `repo/models/resnet18/` (45 MB each, never commit them).

Committed runs and interactive sessions both count against the quota while
the GPU is attached. **Stop an interactive session when you are done**: the
power button at the top right, or *Run → Stop session*. Switch the
accelerator back to *None* for work that needs no GPU.

### Afterwards, on your laptop

* Put the downloaded `_val.csv` files into `models/resnet18/` and run
  `scripts/oof.py` there, or copy the number from the notebook's output.
* Upload `resnet18.csv` on the competition page (*Submit Prediction*), as
  in lab 5.
* Write the row in `EXPERIMENTS.md`, commit and push. The `amp: false` copy
  exists only in the notebook's clone. Mention it in the log row ("Kaggle
  T4, amp off").

<details><summary>No GitHub token? Upload your code as a private Kaggle dataset instead</summary>

On your laptop, zip `src/`, `scripts/` and `configs/` (not `data/`, not
`models/`). On Kaggle: *Datasets → + New Dataset*, drop the zip, keep it
*Private*, *Create*. In the notebook, *+ Add Input → Your Work* adds it. Then,
instead of cell 2:

```python
import glob, os, shutil
code = os.path.dirname(glob.glob("/kaggle/input/**/src/train.py", recursive=True)[0])   # .../src
shutil.copytree(os.path.dirname(code), "/kaggle/working/repo", dirs_exist_ok=True)
os.environ["PYTHONPATH"] = "/kaggle/working/repo"
os.environ["TORCH_HOME"] = "/kaggle/working/.cache/torch"
os.environ["WANDB_MODE"] = "disabled"
%cd /kaggle/working/repo
```

Each change of your code means a new version of that dataset (*New
Version* on the dataset's page). The token route is less work after the
first time.
</details>
