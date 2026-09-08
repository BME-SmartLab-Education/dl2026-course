# Deep Learning 2026 — semester project

Everything you need for the semester project lives here.

| | |
|---|---|
| **[project/](project/README.md)** | What you are building, what data you may use, the rules, and how grading works. **Read this first.** |
| **[practical-01/](practical-01/README.md)** | The photo collection session: what to shoot, how much, and how to upload it |
| **[template/](template/README.md)** | Your starting repository. Docker, PyTorch, and a place for everything |

## In one paragraph

You photograph printed documents, and those photographs become the dataset. Each
one gets a label automatically: how much text an OCR engine loses on it,
expressed as a number between 0 and 1. Over the semester you build models that
predict that number from the image alone, first a classical baseline and then a
convolutional network. The final ranking is a private Kaggle competition on
photographs of documents that appear nowhere in your training data.

## Getting started

1. Read [project/README.md](project/README.md), particularly section 2 on what
   you may use and section 6 on grading.
2. Copy `template/` into a repository of your own and make your first commit.
3. Run `docker compose build` and then the environment check, and confirm your
   GPU is visible before the competition opens.

## The two mistakes that cost the most

**Splitting your data at random.** The same document appears in many
photographs, in two languages. Hold out whole layout groups or your validation
score will lie to you by roughly 11%.

**Augmenting with blur or brightness jitter.** Those change image quality, which
is the thing you are predicting, so they corrupt the label. Flips and small
translations are safe.

Both are explained where they matter, but they are worth knowing before you
write a line of code.
