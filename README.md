# Deep Learning 2026 — semester project

Everything you need for the semester project lives here.

| | |
|---|---|
| **[practices/](practices/README.md)** | What you are building, what data you may use, the rules, and how grading works. **Read this first.** |
| **[practices/001.md](practices/001.md)** | The photo collection session: what was shot, how much, and how it was uploaded |
| **[template/](template/README.md)** | Your starting repository. Docker, PyTorch, and a place for everything |

## Labs

| | |
|---|---|
| **[lab_3_baseline/](lab_3_baseline/README.md)** | Lab 3: the split, a gradient-boosting baseline and a first network. Copied out as your project repository |
| **[lab_4_cnn/](lab_4_cnn/README.md)** | Lab 4: a convolutional network from scratch, the checks before training, the optimiser's starting configuration. Copied *into* your repository from lab 3, as the README describes |

## In one paragraph

You photographed printed documents, and those photographs are the dataset. Each
one has a label computed automatically: how much text an OCR engine loses on
it, expressed as a number between 0 and 1. Over the semester you build models
that predict that number from the image alone, a convolutional network first
and a classical baseline to compare it against. The final ranking is a private
Kaggle competition on photographs of documents that appear nowhere in your
training data.

## Getting started

1. Read [practices/README.md](practices/README.md), particularly section 2 on what
   you may use and section 6 on grading.
2. Copy `template/` into a repository of your own and make your first commit.
3. Run `docker compose build` and then the environment check, and confirm your
   GPU is visible before the competition opens.

## The two mistakes that cost the most

**Splitting your data at random.** The same page appears in dozens of
photographs, in two languages, by the same photographers. Hold out whole units
of the `group_strict` column or your validation score will lie to you by a
fifth.

**Augmenting with blur or brightness jitter.** Those change image quality, which
is the thing you are predicting, so they corrupt the label. Flips and small
translations are safe.

Both are explained where they matter, but they are worth knowing before you
write a line of code.
