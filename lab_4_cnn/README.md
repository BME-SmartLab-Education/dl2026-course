# Lab 4 — convolution: resolution and weight sharing

This folder is not a new project. It is a set of files that you copy **into
your own lab-3 repository** (`dl2026-docphoto`), on top of what is there. Your
repository stays the one repository of the semester; the files in `overlay/`
add a convolutional network, the checks to run before training, the
lecture's starting configuration for the optimiser, and augmentation.

Nothing in `overlay/` overwrites a file that holds one of your lab-3 tasks
or homework (`splits.py`, `mlp.py`, `train.py`, `predict.py`,
`handcrafted.py`, your `README.md`). Three files that were given in lab 3 are
replaced by larger versions of themselves: `src/evaluate.py`,
`src/tracking.py` and `src/models/cnn.py`. If you changed one of those, the
commit in step 1 is how you get your change back.

## Before the lab

Do these at home. In class they cost the hour.

**1. Commit what you have.** In your repository, on your host:

```bash
git status            # must be empty; if not, commit first
git log --oneline -1  # note this commit
```

**2. Copy the overlay in.** From your repository root, with this folder at
`<course>/lab_4_cnn` (wherever you cloned the course repository):

```bash
cp -r <course>/lab_4_cnn/overlay/. .                                   # Linux, macOS, Git Bash
```

```powershell
Copy-Item -Recurse -Force <course>\lab_4_cnn\overlay\* .                # Windows PowerShell
```

**3. Check what changed.** `git status --short` must show exactly these
lines, nothing more and nothing less:

```
 M src/evaluate.py
 M src/models/cnn.py
 M src/tracking.py
?? configs/lab4_cnn_768.yaml
?? configs/lab4_small_cnn.yaml
?? notebooks/lab4_cnn.ipynb
?? src/data/augment.py
?? src/data/inputs.py
?? src/losses.py
?? src/models/build.py
?? src/optim.py
?? src/sanity.py
?? src/train_cnn.py
?? tests/test_lab4.py
```

If another file appears, you copied into the wrong folder; `git checkout .`
and `git clean -fd` undo the copy (they delete untracked files: check the
list first). Then commit: `git add -A && git commit -m "lab 4: start"`.

**4. Build the input tables.** About two minutes, 1.2 GB under `data/work/`:

```bash
docker compose run --rm dev python -m src.data.inputs
```

It must report `grey table 256 px: (1841, 1, 256, 256)` and
`grey table 768 px: (1841, 1, 768, 768)`.

**5. Run the new tests.**

```bash
docker compose run --rm dev pytest tests/test_lab4.py
```

Every test fails with `NotImplementedError: TASK ...`: those are today's
tasks (the `augment` tests belong to the extra TASK 4). Any other error (an
`ImportError`, a missing file) means step 2 or 4
went wrong: fix it before the lab. `pytest tests/test_splits.py` must still
pass.

No new packages: the Docker image of lab 3 has everything.

## Compute

The lab is designed for a laptop **CPU**. Its two core training runs take
about 1 and 3 minutes there (measured with 4 cores; slower laptops take up to
twice as long); you start a run and the discussion happens while it trains.

The notebook has **core** cells, which everyone runs, and **⭐ EXTRA** cells
for GPU owners and for whoever is ahead:

| ⭐ EXTRA | what | CPU | GPU |
|---|---|---|---|
| E1 | TASK 4, augmentation that keeps the label, and its run | 4–8 min | 1 min |
| E2 | the learning-rate range test | 1–2 min | 5 s |
| E3 | the other checks and rungs, including the `ResBlock` task | 25–40 min | 2 min |

**A GPU is not required.** If you have an NVIDIA GPU, enable it as in lab 3:
uncomment the `COMPOSE_FILE` line for your system in `.env`, then
`docker compose down` and `docker compose up lab`. The notebook detects it;
the core runs take seconds, and **every ⭐ EXTRA cell runs by itself**. On a
CPU the extra cells are off; switch one on with `RUN = True` on its first
line if you have the time its heading states. Setting `amp: bf16` in a config
halves the activation memory on a recent GPU.

**On a cluster** with Apptainer instead of Docker, the same image runs
unchanged:

```bash
apptainer build dl2026-student.sif docker-daemon://dl2026-student:latest     # once, on a machine with Docker
apptainer exec --nv --bind $PWD:/workspace --pwd /workspace dl2026-student.sif python -m src.train_cnn --config configs/lab4_cnn_768.yaml
```

## What the files are

| file | what |
|---|---|
| `notebooks/lab4_cnn.ipynb` | the 90-minute lab |
| `src/models/cnn.py` | `SmallCNN` (**TASK 1**), `ResBlock` (**TASK ⭐**), the building blocks |
| `src/sanity.py` | the checks before training: `overfit_one_batch`, `same_seed_twice` (**TASK 2**), memory and cost estimates |
| `src/optim.py` | AdamW with two parameter groups: `param_groups` (**TASK 3**); warmup + cosine schedule |
| `src/data/augment.py` | `augment_safe` (**TASK 4**, ⭐ EXTRA E1); jitter and blur, given, to measure |
| `src/train_cnn.py` | the training loop of the rest of the semester, given; `python -m src.train_cnn --config ...` |
| `src/data/inputs.py` | the input tables, and `preprocess_one` for prediction |
| `src/losses.py` | the loss and how an output becomes a prediction; models return logits from now on |
| `src/models/build.py` | config to model, for training and for `predict.py` |
| `src/evaluate.py` | + `mae`, `page_bootstrap`, `regression_report` |
| `src/tracking.py` | + run groups, figures, tables, confusion matrices in W&B |
| `configs/lab4_small_cnn.yaml`, `configs/lab4_cnn_768.yaml` | the two in-class runs |
| `tests/test_lab4.py` | the checks of the four tasks |

## After the lab

The homework is in the last cell of the notebook. The first item, making
`src/predict.py` work for a CNN checkpoint, is what the next lab's first
Kaggle submission runs on: it has to work before lab 5.
