# Deep Learning 2026 — semester project

**Predict how usable a phone photograph of a document is for automatic text
recognition.**

You build the dataset in the first practical, annotate part of it later in the
term, and spend the semester building models that predict the label. The final
ranking is a private Kaggle competition. Work is **individual**.

---

## 1. The problem, and why it is worth solving

Anyone who has photographed a document with a phone knows that some photos are
fine and some are useless. An automatic system that ingests such photos — an
expense app, a document archive, an insurance claim form, a bank onboarding
flow — has to decide, **at the moment of capture**, whether to accept the shot
or ask the user to take it again.

That timing is the whole point. Asking someone to retake a photo while they are
still holding the phone over the document costs a second. Discovering three
weeks later that the archived image was unreadable costs a support ticket, a
letter, and sometimes the document itself, because the person has thrown the
paper away. The same decision made two seconds earlier is worth orders of
magnitude more than the same decision made later.

So the question is: **given a photograph of a document, how much text will an
OCR engine lose on it?**

That is a regression problem on images, and it has a property that makes it more
interesting than it first looks. The answer is not simply "is this photo
blurry". A reflection over a blank margin costs nothing. The same reflection
over a paragraph destroys it. Two photographs that look equally degraded to any
whole-image statistic can have completely different answers, and separating them
means looking at *where* the damage falls relative to the text.

### "Could I not just ask a vision-language model?"

You will think of it, so let us deal with it now. A large multimodal model can
certainly look at a photograph and say something sensible about it. It is a poor
fit for *this* problem, and the reasons are worth understanding before you start.

**We are measuring, not judging.** The question is not "does this look bad" but
"how many characters does one particular OCR engine lose on it". That depends on
the interaction between a specific degradation and that engine's specific
failure modes. A model can see that a page is dark; it cannot know that this
darkness is precisely where the engine's thresholding gives out. Nothing in a
general-purpose model's training tells it that.

**Reading the page does not help either.** The tempting shortcut is to have the
model transcribe the photo and judge from the result. But the label comes from
comparing the OCR output against the page's true text, which you will not have
at prediction time, and against *that engine's* output, not a better reader's. A
model that reads the page perfectly still cannot say what the OCR would have got
wrong.

**Round numbers lose.** The answer is a value on a continuous scale scored by
RMSE, and the distance between a good solution and a mediocre one is a few
hundredths. Language models tend to emit clustered answers — 0.8, 0.5, 0.9 —
and that habit alone costs more than the margin you are competing over.

**The use case rules it out.** Deciding at the moment of capture means running on
the phone, offline, in milliseconds. A small convolutional network does that. An
API call takes a second or more, needs a network, and costs money per image.

**You cannot debug it.** With your own model you can see where it fails — by
document type, font size, defect — and respond with data or architecture. With a
hosted model you can only rewrite the prompt and hope.

None of which means "do not try it" as an experiment. But note that section 2
forbids any network call at prediction time, so it cannot be part of your
submission. For scale: a stock ResNet-18, trained for about two minutes on a
laptop GPU, reaches a rank correlation of 0.95 with the true labels.

### The target

For each photograph, predict a single number between 0 and 1:

```
usability    = 1 - relative_CER

relative_CER = clip( (CER_photo - CER_baseline_page) / (1 - CER_baseline_page), 0, 1 )
```

`CER` is the character error rate of an OCR engine against the page's known
source text. The page's own baseline CER, measured from a perfect render, is
subtracted so the label describes **the photograph** rather than how hard the
document is to read. 1.0 means the photo costs the OCR nothing; 0.0 means the
text is gone.

**Metric: RMSE** on the private leaderboard.

---

## 2. The data, and what you may do with it

### What you receive

| | |
|---|---|
| **Training images** | The photographs the class collects in Practical 1, from 40 document layouts, each in a Hungarian and an English version |
| **Training labels** | `usability` for every training image, computed automatically by the OCR pipeline |
| **Training metadata** | `page_id`, `layout_id`, `family`, `lang`, `font`, `size_pt` for each training image |
| **Test images** | Photographs of **15 layouts that appear nowhere in the training data**, taken by people not enrolled in the course. No labels, no metadata |
| **Annotation labels** | From Practical 2: human accept/retake judgements on a subset of training images, with per-annotator detail |

All released images have their EXIF metadata stripped, including GPS and device
information.

### What you may use for training

Anything in the table above, plus:

- **Publicly available pretrained weights.** ImageNet backbones, self-supervised
  checkpoints, anything downloadable and documented. Name what you used.
- **The unlabelled test images**, for unsupervised or semi-supervised work:
  pretraining, domain adaptation, pseudo-labelling. You have the pixels; you
  simply have no labels for them.
- **Synthetic data you generate yourself** from the training images.
- **The metadata**, including `layout_id`. You need it to build a correct
  validation split, which is the first thing you should do.

### What you may use at prediction time

> **Your prediction for an image must be a function of that image and your own
> trained weights. Nothing else.**

Allowed, because all of it is computed from the pixels in front of you:

- the pixels themselves, at any resolution
- marker detection and perspective rectification, which you are encouraged to do
- sharpness, exposure, histogram, blockiness and similar statistics, whole-image
  or tile-wise
- an OCR engine **running locally on your machine**, using its output or
  confidence as a feature — legitimate, though it makes inference far slower
  than the real use case would tolerate
- your own trained model, with frozen weights

**Not allowed, and each of these means rejection:**

- **Any network call of any kind.** No web APIs, no hosted models, no
  vision-language or language model services, no cloud OCR, no search, no
  remote inference. Your submission must be producible with the network cable
  unplugged.
- **Any external database, lookup table or corpus** keyed on image content or
  page identity.
- **Any information from other test images.** Your prediction for image X must
  not depend in any way on image Y. See below for why this one matters.
- **The source text of any test page**, obtained or reconstructed by any means.
- **The metadata of any test image** — which layout, which language, which
  font, who photographed it, under what condition.
- **Any label for any test image**, from any source.

### Why the "one image at a time" rule exists

The test set contains several photographs of each page, and you can tell which
photos show the same page because the markers say so. That opens an attack:
combine the OCR output of many degraded photos of one page, take a consensus,
and you have reconstructed something close to that page's true text. With it you
can compute the actual CER of each photo directly and skip modelling entirely.

It is clever, and it is forbidden. It works only because a benchmark hands you
twenty shots of the same document; a phone deciding whether to accept a photo
has exactly one. Exploiting that artefact answers a question nobody asked.

The rule is therefore the simple one: **one image in, one prediction out.**
Unsupervised pretraining on the test images is fine, because that happens while
you are training. Pooling information across test images at prediction time is
not.

---

## 3. What will get your submission rejected

**Using anything at prediction time that is not on the allowed list above.** The
list is exhaustive. If you are unsure whether something counts, ask before the
deadline rather than after.

**Reconstructing the reference text of test pages**, by consensus OCR across
photos of the same page or by any other route.

**Using the condition label.** The training photos are grouped by how they were
taken. That grouping does not exist at test time, and a model trained on it will
have an excellent validation score and a poor leaderboard score.

**Hand-labelling test images.** Looking at test photographs and assigning values
by eye, alone or collectively, is not a model.

**More than one Kaggle account.** One account per student. Extra accounts to
gain submissions or to probe the leaderboard are the clearest form of cheating
here and the easiest to detect.

**Leaderboard probing.** Submissions are limited per day for a reason. Inferring
the test distribution through many targeted submissions — constant predictions,
binary-search probes — is not modelling.

**Sharing predictions, weights or trained models** with other students. The
project is individual. Discussing approaches, papers and debugging is
encouraged; exchanging artefacts is not.

**A submission you cannot reproduce.** If your repository and checkpoint do not
regenerate the predictions you submitted, the submission does not count.

---

## 4. Timeline

| when | what |
|---|---|
| `<<DATE>>` | **Practical 1** — photo collection. Attendance and a valid upload are required |
| `<<DATE>>` | Labelled training set released |
| `<<DATE>>` | **Practical 2** — annotation and consensus labelling |
| **Mon 2 November 2026** | Kaggle competition opens, test images released |
| **Fri 11 December 2026** | Kaggle submissions close, 23:59 |
| **Fri 11 December 2026** | Repository and checkpoint due, 23:59 |
| **Mon 14 – Fri 18 December 2026** | Mark 5 presentations |

The competition runs for **just under six weeks**. That is enough time to build
something good and not enough to leave it until the last weekend, particularly
since the repository has to show that the work happened as you went.

The dates before the competition opens depend on the teaching schedule and will
be announced on the course page.

---

## 5. What you hand in

**Kaggle predictions.** Submitted through the competition page. The private
leaderboard decides the ranking.

**A git repository.** Built on the provided template. It must contain the code
that produced your submission, and someone else must be able to run it. Commit
as you work — a repository whose entire history is one commit on the deadline is
not evidence that you did the work.

**Your trained model checkpoint.** It must be **the model that produced your
submitted predictions**, not a later or a different one. Either:

- commit the checkpoint to the repository, if it is small enough, or
- put a **download link in the repository README**, together with the file's
  size and an SHA-256 checksum, on storage that stays available until the end of
  the exam period.

A repository that runs but has no reachable weights does not satisfy this.

No written report is required. Your repository is the report: if the code is
readable and the experiments are traceable to configs, that is enough.

---

## 6. Grading

### You do not get the signature if any of these is true

- **Your best submission does not beat the reference baseline.** The instructor
  publishes a baseline on the leaderboard, built from classical handcrafted
  image features. A submission scoring worse than it does not qualify, however
  much work went into it. The baseline is beatable by any correctly trained
  network, and it is there to establish that a deep model was actually needed.
- **You have no valid Kaggle submission at all.**
- **You did not contribute photographs** in Practical 1, or your upload was
  unusable. The dataset is a shared resource and everyone contributes to it.
- **You did not take part in the annotation practical.**
- **Your repository does not run**, or does not reproduce the submission you
  made, or the checkpoint behind that submission is not available.
- **You broke one of the rules in sections 2 and 3**, or submitted work that is
  not yours.

### You get the signature if

You contributed photographs, took part in the annotation practical, submitted
predictions that beat the reference baseline, and handed in a repository that
runs together with the checkpoint that produced those predictions.

The signature makes you eligible for the exam. Most students take this route.

### You get a mark 5 and skip the exam

Two ways in.

**By result.** The **top 5 places on the private leaderboard**, among submissions
that satisfy the signature requirements.

**By defence.** Give a **5-minute presentation** on your work and defend it in
questions. To be eligible you must:

- be in the **top 50%** of valid submissions, **and**
- have completed at least **70% of the Slido quizzes** in the lectures

Eligibility is not the standard. The presentation has to show something worth
hearing: an experiment that answers a real question, an error analysis that
found something non-obvious, a method that goes beyond following the template.
Reproducing the baseline competently earns a signature, not a five.

Good material for a defence:

- **The human labels against the machine labels.** Practical 2 gives you
  accept/retake judgements from five annotators per image. The Kaggle
  competition does not score them, which leaves the interesting question open:
  where do people and the OCR engine disagree, and is the disagreement
  structured? Two effects are worth hunting for — a steeply angled photo looks
  bad to a person but rectifies away to nothing for the OCR, while a mild blur
  nobody notices destroys 7.5 point text. Predicting the human label, or the
  annotator disagreement itself, is a genuine modelling problem on data you
  helped create.
- **The cost of the wrong validation split**, measured on your own data rather
  than quoted from this document.
- **What your model gets wrong**, sliced by document family, font size and
  defect type, and what you did about it.
- **An ablation that isolates why something worked**, rather than a leaderboard
  score that moved for unclear reasons.

If you want this route, register your intent by **Friday 11 December**, the same
deadline as everything else, so the week can be scheduled. Presentations are
held between **14 and 18 December**.

### Everyone else

Signature, then the exam.

---

## 7. Two things that will cost you if you ignore them

**Split by layout, not at random.** The same document appears in many photos, and
every layout exists in two languages typeset identically. A random split puts
near-identical photographs on both sides and your validation score becomes
optimistic — we measured about **11%** on simulated data. Hold out whole layout
groups. The template has a place for this and a test that checks it.

**Do not augment with anything that changes image quality.** Flips and small
translations are safe. Blur, brightness jitter and added noise change the very
thing you are predicting, so they corrupt the label. This is the most common way
to make a model quietly worse while believing you improved it.
