# Document photo quality — semester project

Predict how usable a phone photograph of a printed document is for automatic
text recognition. One number per image, between 0 and 1.

```
usability = 1 - relative_CER
```

1.0 means the photograph costs the OCR nothing. 0.0 means the text is gone. The
competition metric is RMSE.

This repository is your starting point. The infrastructure works; the modelling
code is yours to write. Every source file is a stub saying what belongs in it.

---

## Setup

The only thing you need installed is **Docker**. Python, PyTorch, CUDA and every
library live in the image, so the project behaves the same on your laptop, on a
lab machine and on ours.

```bash
git init
git add .
git commit -m "Initial project structure"

docker compose build
docker compose run --rm dev python scripts/check_env.py
```

The second command should end with a GPU line and a successful matmul. If it
says the GPU is unavailable, everything still runs, just slowly.

```
python       3.11.9 on Linux
torch        2.4.1+cu124
gpu          NVIDIA RTX 3500 Ada Generation Laptop GPU (12.0 GB)
gpu check    matmul ok
project root /workspace
```

If your machine has an NVIDIA GPU, add the overlay so the container can see it:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev python scripts/check_env.py
```

Then put the data you were given in `data/raw/` and leave it alone.

### Experiment tracking

Runs are logged to **Weights & Biases**. Set it up once:

```bash
cp .env.example .env      # then paste your key from https://wandb.ai/authorize
```

Docker Compose reads `.env` automatically and passes the key into the container.
`.env` is gitignored — **never commit your key**, and if you paste one into a
notebook by accident, revoke it rather than deleting the line.

Three modes, set by `WANDB_MODE` in `.env`:

| value | what it does |
|---|---|
| `online` | normal, logs to wandb.ai |
| `offline` | logs to disk; upload later with `wandb sync wandb/…` |
| `disabled` | no logging at all, useful while debugging |

You will run dozens of experiments this semester and you will not remember which
one produced which number. That is the problem this solves. Log the resolved
config, the loss curves, the final metrics, and the checkpoint as an artifact —
the last one matters, because you have to hand in the exact weights that
produced your submission.

### Everyday commands

```bash
docker compose run --rm dev bash                    # interactive shell
docker compose run --rm dev pytest                  # tests
docker compose run --rm dev python -m src.train --config configs/cnn.yaml
docker compose up lab                               # Jupyter Lab, localhost:8888
```

These are long to type a hundred times, so define an alias once:

```bash
# bash / zsh, in your shell profile
alias dc='docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev'
dc python -m src.train --config configs/cnn.yaml
```

```powershell
# PowerShell, in $PROFILE
function dc { docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm dev @args }
dc python -m src.train --config configs/cnn.yaml
```

There is deliberately no Makefile. The commands above *are* the interface, and
when something breaks you will be debugging Docker rather than a wrapper around
it.

### How the container is put together

Three details are worth understanding, because they are the ones that break
when you change something.

**The project is mounted, not copied.** Editing a file on your host changes it
in the container immediately. You only need `docker compose build` again when
`requirements.txt` or the `Dockerfile` changes.

**Torch comes from the base image and is deliberately not in
`requirements.txt`.** Installing it again would fetch a second, possibly
CPU-only build and quietly waste your GPU.

**The container runs as a normal user, not root**, so the files it writes into
your project are yours. If your host user id is not 1000 you will see permission
errors; rebuild with `docker compose build --build-arg UID=$(id -u)`.

The GPU is enabled by adding `-f docker-compose.gpu.yml` to the compose command,
as shown above. Leave it out on a machine without an NVIDIA GPU and everything
still works, just slowly.

---

## Layout

```
src/
  config.py            paths and settings everything else imports
  data/
    dataset.py         loading images and labels
    splits.py          train and validation splits   <- write this first
  features/
    handcrafted.py     classical image quality features
  models/
    baseline.py        features + a classical regressor
    cnn.py             convolutional models
  utils/
    seed.py            reproducibility
    logging.py         run logging
  train.py             one entry point, driven by a config
  evaluate.py          scoring and error breakdown
  predict.py           writes a submission file  <- read the contract in it

configs/               one YAML per experiment
.env.example           copy to .env, holds your W&B key (never committed)
scripts/check_env.py   verifies the container, prints versions and GPU status
notebooks/             exploration; move anything reusable into src/
tests/                 start with test_splits.py
data/                  raw / interim / processed, none of it tracked
models/                checkpoints, not tracked
runs/                  local run artefacts and figures (W&B holds the metrics)
submissions/           competition files
```

## Making a Kaggle submission

One command produces one file:

```bash
docker compose run --rm dev python -m src.predict \
    --config configs/cnn.yaml \
    --checkpoint models/cnn_best.pt \
    --out submissions/cnn_best.csv
```

Then upload `submissions/cnn_best.csv` to the competition page.

### The format

Exactly two columns, one row per test image, in the order given by the
competition's `sample_submission.csv`:

```csv
image_id,usability
test_0001,0.8213
test_0002,0.1044
```

`image_id` is the test filename without its extension, copied from the sample
submission. `usability` is a float in [0, 1] — clip it, because a prediction of
1.3 is not a bolder guess, it is an error the metric charges you for.

`src/predict.py` builds the output *from the sample submission* rather than from
a directory listing, so the ids and the row order cannot drift. It also asserts
four things before writing, each of which has cost somebody a submission slot:
the row count matches, ids are unique, there are no NaNs, and every value is
inside [0, 1].

### One image in, one prediction out

Your prediction for an image must depend on **that image and your trained
weights only**. That is why the loop in `predict.py` handles one image at a time
and shares nothing between iterations.

Batching for speed is fine — a batch of independent images still gives each its
own answer. What is not allowed is anything that pools information across the
test set: normalising by statistics computed over all test images, clustering
photos of the same page, or reconstructing a page's text by combining several
photos of it. Read section 2 of the project description before you get clever
here; that section exists because one of those tricks would work.

### The checkpoint is part of the submission

You hand in the weights that produced your predictions, not a later or better
version. Log every checkpoint to W&B as an artifact of the run that made it and
this question answers itself; otherwise keep a note in the repository saying
which file goes with which submission.

## The order of work

1. **Write the split.** Before any model. See below.
2. **Look at the data.** Plot the label distribution. Compare the best and worst
   photos and work out what distinguishes them.
3. **Build the baseline.** Handcrafted features plus a classical regressor. This
   is the number a deep model has to beat to justify itself.
4. **Build a CNN.** Pretrained backbone, regression head. Compare honestly.
5. **Improve it.** Resolution, augmentation, architecture, ensembling.

## Write the split first

The same document layout appears in many photographs, and every layout exists in
both a Hungarian and an English version. If you split rows at random,
near-identical photos of the same page land in training and validation at the
same time, and the score you report is optimistic.

We measured the effect on simulated data: **a naive per-image split flatters
validation RMSE by about 11 %.**

Hold out whole layout groups, keep both language variants of a layout on the
same side, and keep the family mix similar in both. `tests/test_splits.py` is
there for you to assert that no group appears on both sides. It is a few lines,
and it protects every number you report this semester.

Implementing the random split too and measuring your own gap makes a far more
convincing paragraph in your report than a number we gave you.

## Things that will cost you if you get them wrong

**Augmentation that changes image quality.** Flips and small translations are
safe. Blur, brightness jitter and added noise change the very thing you are
predicting, so they corrupt the label. This is the most common way to make a
model quietly worse.

**Leaking the photo condition.** The training filenames record how each photo
was taken. That information does not exist at test time. Train on it and your
validation score will be excellent and your leaderboard score will not.

**Reporting a single number.** RMSE tells you how well you did, not what to fix.
Break the error down by document family, language, font size and defect type.

**Not fixing the seed.** You will compare dozens of runs. A difference smaller
than your run-to-run variance is not a result.

**Committing data or checkpoints.** `.gitignore` covers it, but check
`git status` before your first commit rather than after.

## Notes on the data

Photographs vary in perspective, lighting, background surface and phone. Because
the pages carry corner markers and are rectified before OCR, the angle matters
much less than it looks: a 47 degree shot scores about the same as a straight-on
one.

The defects that matter most are the **local** ones, a reflection or a shadow
across part of the page. The same defect over one column and over empty margin
looks nearly identical to a whole-image statistic and produces very different
labels. That is the core difficulty of this task, and it is why an average of
sharpness or brightness over the frame only gets you so far.
