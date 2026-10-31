# The reference solution

This is the model behind the **Reference solution** line on the competition
leaderboard, and the repository you start the competition from. It is lab 4's
network with lab 4's best setting, trained on all the training photographs: a
small convolutional network of 25 thousand weights, trained from scratch on
grey 768 × 768 px pages. About four hundred lines of Python.

For the signature you improve on it in at least four validated steps, and
your selected submission must reach a private RMSE at least 25 % below the
reference's, that is **0.157 or lower** (`practices/README.md`, section 6).

| RMSE | 4-fold CV by `group_strict` | public leaderboard | private leaderboard |
|---|---|---|---|
| constant (the training mean, 0.748) | 0.230 | 0.280 | 0.275 |
| **this reference** | **0.194** | **0.216** | **0.210** |
| the signature line | | | 0.157 |

**The leaderboard numbers are higher than the cross-validation ones, for every
model.** The test pages are new layouts photographed with other phones in other
rooms, and their labels are more spread out: even the constant does worse
there. Compare your models with each other on the folds; the leaderboard tells
you where you stand, not which of two close variants is better.

One run takes about 2 minutes on a laptop GPU and about 20 minutes on
**four laptop CPU cores**, so the reference itself needs no GPU.
What you build on it probably will (see [below](#without-a-gpu-of-your-own)).

## Set it up

This folder is a repository of its own, with lab 3's Docker image. It does not
go into your lab-3 repository: copy it out of the course clone and start a new
repository from it.

```bash
git -C <course> pull                                           # Linux, macOS, Git Bash
cp -r <course>/reference ~/dl2026-competition
cd ~/dl2026-competition
git init && git add . && git commit -m "reference solution"
```

```powershell
git -C <course> pull                                           # Windows PowerShell
Copy-Item -Recurse <course>\reference $HOME\dl2026-competition
cd $HOME\dl2026-competition
git init; git add .; git commit -m "reference solution"
```

Commit it as it is before you change anything, so your own changes show up as
diffs against the reference. Push it to a **private** repository: rule 7 asks
for the code behind your submission.

**Docker**, as in lab 3:

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
docker compose build          # instant if lab 3's image is still there: same name, same content
docker compose run --rm dev python scripts/check_env.py
```

On a machine with an NVIDIA GPU, uncomment the `COMPOSE_FILE` line for your
operating system in `.env`. The reference does not log to Weights & Biases;
the key in `.env` is there for what you add.

**The data**, four things in `data/raw/`, exactly there:

```
data/raw/train.csv                the training dataset
data/raw/train/                   1841 .jpg
data/raw/test/                    1628 .jpg, from the competition's Data tab
data/raw/sample_submission.csv    from the competition's Data tab
```

Copy `train.csv` and `train/` from your lab-3 repository's `data/raw/`, or
download the training dataset again (the link is on Moodle). Download the
competition's data from its *Data* tab (*Download All*) and unpack it into
`data/raw/` so that `test/` and `sample_submission.csv` land next to
`train/`. Nothing in `data/` goes into git; `.gitignore` keeps it out.

## Run it

Three commands, from the repository root:

```bash
docker compose run --rm dev python -m src.prepare                   # resize once, ~3 min
docker compose run --rm dev python -m src.train --fold all          # train on every photograph
docker compose run --rm dev python -m src.predict models/reference/smallcnn_768_foldall_seed42.pt
```

The last one writes `submissions/reference.csv`. Check it before you upload
it:

```bash
docker compose run --rm dev python scripts/check_submission.py submissions/reference.csv
```

Its score will be close to the reference line but probably not equal to it:
another seed moves the private score anywhere between 0.204 and 0.216, and
another GPU or a CPU does the arithmetic in a slightly different order.

## Validate on the folds

The leaderboard is no place to compare two versions of a model: five
submissions a day, a public third of the test pages, and differences of a few
thousandths that are noise there. The folds are. Train on three folds and
score the fourth:

```bash
docker compose run --rm dev python -m src.train --fold 0             # folds 1-3 train, fold 0 validates
```

It prints the validation RMSE after every epoch and writes the predictions to
`models/reference/*_val.csv`. Run all four (`--fold 0`, `1`, `2`, `3`); the
number to compare is the RMSE of the four validation files pooled together,
the out-of-fold RMSE, which is 0.194 for the reference. The fold split is by
`group_strict` (lab 3): the six strict units of the training set are dealt
into 4 folds of 370 to 560 photographs, and no page or layout is on both
sides.

Everything that changes the result is in `configs/reference.yaml`. To try
something, copy it (`cp configs/reference.yaml configs/my_try.yaml`), change
the copy, and pass `--config configs/my_try.yaml` to `train`. Keep every
config you submitted from; rule 7 asks you to reproduce the submission.

## What is in it, and why

| file | what |
|---|---|
| `configs/reference.yaml` | data paths, image size, the network's widths, the training schedule |
| `src/common.py` | the folds, the resize, the letterbox, the augmentation, the network |
| `src/prepare.py` | resizes every photograph once to 1024 px on the long side |
| `src/train.py` | one training run, on three folds or on all of them |
| `src/predict.py` | one or more checkpoints in, a submission out |
| `scripts/check_submission.py` | checks a submission file before you upload it |
| `scripts/check_env.py` | checks the container, as in lab 3 |
| `EXPERIMENTS.md` | the template of your experiment log |
| `Dockerfile`, `docker-compose*.yml`, `requirements.txt` | lab 3's environment, unchanged |

The choices, and the lab each one comes from:

* **Validation by `group_strict`** (lab 3). A random split would put
  photographs of the same page on both sides and flatter the score by a fifth.
* **Grey pages at 768 px through a 7 × 7 stride-4 stem** (lab 4). The label is
  mostly about fine blur; at 256 px it is gone. The stem reads every pixel
  and hands a 192 px map to the rest of the network. The resize uses area
  averaging for the same reason.
* **Conv-BatchNorm-ReLU blocks, average and maximum pooling side by side**
  (lab 4). The average says how sharp the page is overall, the maximum how
  bad its worst region is.
* **He initialisation, the output bias at the training mean** (lecture 4,
  lab 4). The network returns a logit; the sigmoid is applied in the loss and
  in `predict`.
* **AdamW with weight decay on the weights only, warm-up then cosine, gradient
  clipping, 12 epochs, the last epoch kept** (lecture 5, lab 4). The number of
  epochs is fixed in advance, so the `all` run needs no validation set and a
  fold run measures exactly what the `all` run does.
* **Mirror and shift only** (lab 4). Brightness and blur are what the label
  measures; augmenting them trains the network on wrong labels.
* **MSE loss**, because the metric is RMSE.

## Where to go from here

The reference is where lab 4 ended. Everything since then is the map:
lecture 5 (optimisers, the learning-rate range test, regression losses),
lecture 6 (architectures, transfer learning, augmentation), lab 5 (pretrained
networks) and lab 6 (ensembles). None of it is in the reference, and much of
it matters on this data. Which parts, and how much, is what your experiments
are for. Code from your lab repositories may be brought over; this repository
is the one you hand in.

Two habits make the difference between four steps and four guesses:

* **One change at a time, four folds each.** A change that does not lower the
  out-of-fold RMSE does not help, however reasonable it sounds. Small
  differences on a single fold are noise.
* **Look at the errors.** The `*_val.csv` files have `family`. Where is the
  model worst, and why?

## The experiment log

The signature asks for at least four changes, each of which lowered the
out-of-fold RMSE compared with the step before it. Record them in
`EXPERIMENTS.md` at the root of this repository, as you go, not on the
deadline. Changes that did not help belong in it too; they just do not count
towards the four.

## Without a GPU of your own

**Kaggle Notebooks** give 30 GPU hours a week for free (*Settings → Accelerator
→ GPU*; your phone number has to be verified). In a new notebook, add the
competition and the training dataset as inputs (*Add Input*), switch
*Internet* on (for `git clone` and any pretrained weights), then:

```python
!git clone https://<token>@github.com/<you>/dl2026-competition.git repo
%cd repo
!ln -s /kaggle/input/<training-dataset>/train data/raw/train
!cp /kaggle/input/<training-dataset>/train.csv data/raw/
!ln -s /kaggle/input/<competition>/test data/raw/test
!cp /kaggle/input/<competition>/sample_submission.csv data/raw/
!python -m src.prepare && python -m src.train --fold all
!python -m src.predict models/reference/smallcnn_768_foldall_seed42.pt
```

Look up the two `/kaggle/input/...` folder names in the notebook's input panel.
A private repository needs a GitHub token with read access in the URL (add it
as a Kaggle *Secret*, never in the notebook's text). Download the checkpoint
and the submission from the *Output* panel and commit the config you trained
with; the checkpoint goes where `practices/README.md` section 5 says.
**Colab** works the same way with the data copied to a private Google Drive.
Either way the photographs stay private: never a public notebook, never a
public link.
