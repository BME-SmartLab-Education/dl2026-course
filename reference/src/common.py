"""What training and prediction share: the config, the folds, the images, the network.

Anything that touches pixels lives here and nowhere else. If training and
prediction prepared an image in two slightly different ways, the difference
would not show in the validation score and would show on the leaderboard.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
TARGET = "usability"
GREY = 128


def load_config(path: str | Path = ROOT / "configs" / "reference.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(dev: torch.device, enabled: bool = True):
    """bfloat16 on a GPU that has it (Ampere and later); full precision everywhere else."""
    on = enabled and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=on)


# ------------------------------------------------------------------ tables

def train_table(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["data"]["raw"]) / "train.csv")


def test_table(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["data"]["raw"]) / "sample_submission.csv")[["image_id"]]


def assign_folds(df: pd.DataFrame, column: str, n_folds: int) -> np.ndarray:
    """A fold number per row; every unit of `column` lands whole in one fold.

    Units are dealt largest first to the fold with the fewest photographs so
    far, which keeps the folds close in size. Deterministic: the same table
    gives the same folds on every machine.
    """
    sizes = df[column].value_counts().sort_index()
    order = sorted(sizes.index, key=lambda u: (-sizes[u], u))
    load, fold_of = [0] * n_folds, {}
    for u in order:
        k = int(np.argmin(load))
        fold_of[u] = k
        load[k] += sizes[u]
    return df[column].map(fold_of).to_numpy()


# ------------------------------------------------------------------ images

def cache_path(cfg: dict, split: str, image_id: str) -> Path:
    return Path(cfg["data"]["work"]) / f"{split}_{cfg['data']['cache_size']}" / f"{image_id}.jpg"


def resize_photo(src: Path, dst: Path, size: int) -> None:
    """Long side to `size` px, aspect kept. Used for training and test photographs alike."""
    im = Image.open(src)
    im.draft("RGB", (size, size))          # decode the JPEG at 1/2 or 1/4 scale when that is enough
    im = im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=95)


def letterbox(path: Path, size: int) -> np.ndarray:
    """The photograph as grey uint8 (size, size): long side scaled to `size`, centred on mid-grey.

    Area averaging when shrinking, so that fine blur survives the resize as
    well as it can; that blur is what the label is mostly about. Grey, as in
    labs 3 and 4: the cache keeps the colour for whoever needs it.
    """
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    s = size / max(h, w)
    if s != 1.0:
        im = cv2.resize(im, (round(w * s), round(h * s)),
                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = im.shape
    out = np.full((size, size), GREY, np.uint8)
    top, left = (size - h) // 2, (size - w) // 2
    out[top:top + h, left:left + w] = im
    return out


class Photos(Dataset):
    """Photographs from the resize cache, letterboxed, as float tensors (1, size, size) in [0, 1].

    With `augment`, each photograph is mirrored left-right with probability
    1/2 and shifted by up to `max_shift` px (lab 4's "safe" augmentation).
    Nothing else: brightness, contrast and blur are what the label measures,
    so changing them would make a training example with the wrong label.
    """

    def __init__(self, paths, targets=None, size: int = 768, augment: bool = False,
                 max_shift: int = 16):
        self.paths = list(paths)
        self.targets = None if targets is None else np.asarray(targets, np.float32)
        self.size, self.augment, self.max_shift = size, augment, max_shift

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = letterbox(self.paths[i], self.size)
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            s = self.max_shift
            if s:
                pad = np.pad(x, s, constant_values=GREY)
                dy, dx = np.random.randint(0, 2 * s + 1, size=2)
                x = pad[dy:dy + self.size, dx:dx + self.size]
        x = torch.from_numpy(np.ascontiguousarray(x)[None]).float().div_(255)
        y = torch.tensor(self.targets[i]) if self.targets is not None else torch.tensor(float("nan"))
        return x, y


def worker_init(worker_id: int) -> None:
    # each DataLoader worker gets its own numpy stream, derived from torch's seed
    np.random.seed(torch.initial_seed() % 2**32)


# ------------------------------------------------------------------ network

def conv_block(c_in: int, c_out: int, stride: int = 1, kernel: int = 3) -> nn.Sequential:
    """Convolution, BatchNorm, ReLU. No bias in the convolution: the norm would subtract it again."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Lab 4's network: a strided stem, conv-pool blocks, average and maximum pooling, one logit.

    The 7 x 7 stride-4 stem reads every pixel of the 768 px page and hands a
    192 px map to the blocks; that is what made 768 px affordable in lab 4.
    The pooling concatenates the mean over all positions ("how sharp is the
    page overall") with the maximum ("how strong is the strongest response
    anywhere"). The forward pass returns a logit; `predict` and the loss in
    train.py apply the sigmoid.

    The network standardises its own input with the mean and standard
    deviation of the training pixels, stored in the checkpoint, so that
    prediction cannot forget to do it.
    """

    def __init__(self, widths=(8, 16, 32, 64), stem_stride: int = 4, stem_kernel: int = 7,
                 dropout: float = 0.0, prior: float = 0.5):
        super().__init__()
        layers = [conv_block(1, widths[0], stride=stem_stride, kernel=stem_kernel)]
        for c_in, c_out in zip(widths[:-1], widths[1:], strict=True):
            layers += [conv_block(c_in, c_out), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * widths[-1], 1)
        self.register_buffer("in_mean", torch.tensor(0.5))
        self.register_buffer("in_std", torch.tensor(0.25))

        # He initialisation for the ReLU stack (lecture 4); the head starts
        # with tiny weights and its bias at logit(prior), so that step 0
        # predicts the training mean instead of 0.5.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        p = min(max(float(prior), 1e-3), 1 - 1e-3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(p / (1 - p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features((x - self.in_mean) / self.in_std)
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        return self.head(self.drop(f)).squeeze(1)


def build_model(cfg: dict, prior: float = 0.5) -> SmallCNN:
    m = cfg["model"]
    return SmallCNN(m["widths"], m["stem_stride"], m["stem_kernel"], m["dropout"], prior)


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    return model, cfg


@torch.no_grad()
def predict(model: nn.Module, loader, amp: bool = True) -> np.ndarray:
    """Usability in [0, 1] for every photograph of the loader, in order."""
    dev = device()
    model.to(dev).eval()
    out = []
    for x, _ in loader:
        with autocast(dev, amp):
            out.append(torch.sigmoid(model(x.to(dev, non_blocking=True)).float()).cpu())
    return torch.cat(out).numpy()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
