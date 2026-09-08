# Practical 1 — Building the dataset

**Deep Learning, 2026 autumn. Bring your phone.**

Today you create the data the whole semester runs on. There is nothing to
download: the dataset does not exist yet, and by the end of this session you
will have built it.

---

## 1. What you are building

An automatic system that ingests phone photos of documents — an expense app, a
document archive, an insurance claim — has to decide at the moment of capture
whether to accept a shot or ask for another. Asking again while the person is
still holding the phone over the paper costs a second. Finding out three weeks
later that the archive is unreadable costs far more.

So the semester project is a model that answers: **given a photograph of a
document, how much text will an OCR engine lose on it?** The full motivation,
the rules and the grading are in the project description; today is about the
data.

The part that matters for your photography is this. The answer is not simply
"is this photo blurry". A reflection over a blank margin costs nothing; the same
reflection over a paragraph destroys the text. Photographs that look equally bad
to a whole-image measurement can have completely different answers, and that
distinction is what makes the problem worth studying. It is also the thing only
you can put into the dataset, which is why section 6 keeps insisting on the word
*partial*.

## 2. The target

For each photograph you predict a single number between 0 and 1:

```
usability    = 1 - relative_CER

relative_CER = clip( (CER_photo - CER_baseline_page) / (1 - CER_baseline_page), 0, 1 )
```

**CER** is the character error rate: the edit distance between what the OCR
engine reads and the true text of the page, divided by the length of the true
text. A CER of 0 means a perfect transcription; 1 means nothing usable came out.

**Why the baseline is subtracted.** Documents differ in how hard they are to
read even under ideal conditions. A dense two-column legal text set in 7.5 point
condensed type produces some OCR errors from a flawless scan; a short
certificate in 13 point serif produces none. If we used raw CER, the label would
mostly encode *which document* is in the photo, and a model could score well by
recognising the layout and ignoring the photography entirely.

So for every page we measure `CER_baseline_page` once, from a perfect render of
that page, and subtract it. What remains is attributable to your photograph. The
denominator `(1 - CER_baseline_page)` rescales the remainder back onto [0, 1].

**Reading the number.** 1.0 means the photo costs the OCR nothing. Around 0.5
means roughly half the recoverable text is gone. 0.0 means the text is gone
entirely.

## 3. How the label is produced

You will not annotate anything. Every photograph is labelled automatically by a
pipeline, and it is worth understanding it, because two of its properties
determine what makes a photo useful and what makes it worthless.

Each page carries **four ArUco markers**, one in each corner. They do two jobs.

**They identify the page.** Each marker encodes an integer, and the four on a
page are `4 x page_id + corner`. The pipeline detects the markers, divides by
four, and knows exactly which document it is looking at — and therefore which
source text to compare against. This is why you never have to record which page
you photographed.

**They define the geometry.** The inner corners of the four markers are four
known points on a sheet of known size. That is enough to compute a perspective
transform that flattens your photo back to a true, rectangular A4 page before
anything else happens. A photo taken from the side is straightened out
automatically.

The full chain, per photograph:

1. **Detect** the four markers. If fewer than four are found, stop: this image
   cannot be used at all.
2. **Identify** the page from the marker IDs.
3. **Rectify** the sheet to a flat A4 at 200 dpi using the marker corners.
4. **OCR** the rectified page with Tesseract, in the correct language, since the
   pages exist in Hungarian and English versions.
5. **Compare** the output with the known source text after normalisation, giving
   the CER.
6. **Subtract** that page's baseline and clip, giving the usability label.

Two consequences follow directly, and they are the two things to remember:

> **A photo whose markers cannot be found produces no label at all.** It is not
> a hard training example, it is a discarded file. This is the one rule that
> never relaxes.

> **Angle costs you almost nothing.** Because step 3 undoes perspective, we
> measured a 47 degree shot scoring the same as a straight-down one, and marker
> detection surviving past 60 degrees. Shoot from wherever is natural.

## 4. What we do with it afterwards

The labelled photographs become the training set for the semester project: a
model that predicts usability from an image. You will build a classical
baseline from handcrafted image features first, then a convolutional network,
and compare them honestly.

The competition test set is photographed separately, from **fifteen document
layouts that are not in today's pool at all**, by people who are not taking
part. So a model that memorises the documents in front of you today will score
well on your own validation split and badly on the leaderboard. Which brings us
to the one methodological warning worth giving in advance.

**Do not split your data randomly by image.** The same layout appears in many
photographs, and every layout exists in both a Hungarian and an English version,
typeset identically. A random split puts near-identical photos of the same page
into training and validation simultaneously, and your validation score becomes
optimistic. We measured the effect on simulated data: **about 11 %**. Hold out
whole layout groups instead. The project template has a place for this and a
test that checks it.

---

## 5. Session plan

| time | what |
|---|---|
| 0:00–0:15 | The task, the label, the semester |
| 0:15–0:30 | How the markers and the labelling pipeline work |
| 0:30–0:40 | The shot list, with a live demonstration of each condition |
| 0:40–1:15 | You shoot |
| 1:15–1:30 | Upload, automatic marker check, retakes |

Pages are laid out on the tables. Move between tables as you work, and do not
take all your photos from the nearest one — every page needs coverage.

---

## 6. Your assignment: 50 photographs

Use your own phone, default camera settings. Install nothing.

**Estimated time: 45 to 60 minutes.**

### Part 1 — 15 clean reference photographs

Fifteen different pages, one photo each. For each:

- the page lying flat on the table
- filling at least two thirds of the frame
- all four corner markers visible
- sharp, no camera shake
- even light, no shadow across the page
- nothing else in frame: no other papers, no hands, no screens, no people

These anchor the top of the scale. Boring and correct is exactly right.

### Part 2 — 35 degraded photographs

Now introduce problems on purpose. Spread across as many different pages as you
can, and **make about ten of them pages you already photographed in Part 1**, on
the same phone. Those pairs let us compare a real defect against a simulated one
on identical material, which is something we cannot get any other way.

| how many | condition | what to do |
|---|---|---|
| **12** | **partial shadow** | Let your hand, head or body throw a shadow across **part** of the page, so some text is shadowed and some is not |
| **9** | **partial reflection** | Angle the page so a lamp or window reflects off **part** of the paper, washing out some of the text |
| **8** | **motion** | Move the phone as you press the shutter. Vary it: some barely smeared, some heavily |
| **4** | **low light** | Turn away from the light or step into a dark corner. No flash |
| **2** | **however an impatient person would do it** | Your choice |

### The single most important word: *partial*

For shadow and reflection, the instruction is **part of the page**, not all of
it.

A shadow over the whole sheet is just a dark photograph, and the simplest
possible brightness measurement detects it. A shadow over **one column**, with
the rest of the page perfectly lit, is a completely different problem: a model
has to work out where the damage is and whether it covers text or empty margin.
That distinction is the entire reason this task needs a neural network rather
than a handful of image statistics, and it is the part only you can produce.

We measured how much this matters. The same strength of shadow scores **0.82
when it falls on empty margin and 0.47 when it falls on text**. Those two
photographs look nearly identical to a whole-image average.

### How degraded is degraded enough?

Aim for the **middle**, not for ruined.

> The photo should still look like a readable document at a glance, but you
> would not want to hand it in.

A barely-degraded photo scores above 0.9 about nine times out of ten, and a
completely wrecked one scores near 0. Both are easy for any model and carry
little information. The photographs that decide the competition are the ones in
between, and they are the hardest ones to produce on purpose, which is exactly
why we are asking people rather than simulating them.

### Why the quotas are quotas and not suggestions

A model learns nothing about a kind of damage it never sees. We tested this by
training with one defect type removed: on photographs showing that defect, the
model scored **worse than guessing the average**, while remaining perfectly
accurate on everything else. A missing condition is not a small weakness spread
thinly, it is a hole.

So if the class collectively produces very few reflection photos, every model
built this semester will be bad at reflections, and there is no way to repair it
afterwards. Please hit the numbers roughly, even if one condition is more
awkward to set up than another.

### Three notes on the conditions

**Shadow is the most valuable and the easiest to get right.** It reaches the
useful middle of the range with the least effort — just put your hand between
the lamp and part of the page. That is why it has the largest quota.

**Low light has a small quota on purpose.** Modern phones compensate for dim
conditions remarkably well, so a "dark" photo is usually either fine or, past a
certain point, completely destroyed, with very little in between. Three is
enough to cover it.

**Angle is not a degradation.** The rectification removes it, so a tilted photo
is not a bad photo. Vary your angle freely across *all fifty* shots for realism,
but do not spend any of your quota trying to make a photo bad by tilting it.

---

## 7. Rules

- **All four corner markers visible in every photo.** No exceptions.
- **No editing.** No filters, no cropping, no rotation, no "enhance". Turn off
  HDR if you can. Upload exactly what the camera saved.
- **Do not retake a photo to improve it.** A mediocre photo is data. Delete only
  genuine mistakes, such as a shot of the table.
- **Do not photograph a screen.** Only the printed sheets.
- **Do not flatten or iron the pages.** Ordinary paper curl is realistic.
- **Keep your phone's default resolution.** Do not downscale.

## 8. Upload

Put your fifty photos in a **single ZIP file** named with your Neptun code
(`ABC123.zip`) and attach it to the assignment in the course's **Microsoft Teams
group**:

> **Teams assignment:** `<<ASSIGNMENT_NAME>>`
>
> **Deadline:** `<<DEADLINE>>`

Teams already knows who you are, so the Neptun code in the filename is only
there to make the files easy to match against the class list once they are
downloaded. Nobody else can see your submission.

Two practical notes. **Upload from the Teams desktop app rather than the
browser** — the web client is unreliable with attachments of this size. And
**press Hand in**, not just Attach: a file that is attached but not handed in
does not count as submitted.

Inside the ZIP, put the Part 2 photos in folders named after the condition:

```
your_photos.zip
├── clean/          (15 photos)
├── shadow/         (12)
├── reflection/     (9)
├── motion/         (8)
├── lowlight/       (4)
└── free/           (2)
```

**No filename rules inside the ZIP.** The markers identify the page, so nothing
needs to be tracked by hand. The folders are the only sorting you do, and they
are far harder to get wrong than a filename convention.

**Do not agonise over which folder a photo belongs in.** How degraded a photo
actually is gets measured by the OCR pipeline, not by where you filed it. If a
shot you meant to be clean came out shaky, it will be scored as a shaky photo
and used as one. The folders exist so we can check on the day that the class has
produced enough of each condition, and so we can report results by defect type
later. They are not a grade.

The condition folder is **not** given to you afterwards. During the competition
you receive only the image.

We run the marker check while you are still in the room. If some of your photos
have lost a corner you will be told, and you can retake them on the spot.

## 9. Privacy

Every document is **synthetic**. The names, addresses, company details and
figures are randomly generated and refer to no real person or organisation.
Nothing printed on these pages is personal data.

Do not photograph anything else — no faces, no other documents, no screens. If
something of yours appears in the background of a shot, delete it and retake.

We strip EXIF metadata from every image before the dataset is released,
including GPS coordinates and device information. Device details are kept only
in our own records, for checking that the collection covers a range of phones;
they are not part of the data anyone trains on.

## 10. Common mistakes

- **Photographing only nice pictures.** The dataset then has no range, every
  model looks equally good, and the competition is decided by chance.
- **Making the degraded photos so bad the text is gone entirely.** Aim for the
  middle.
- **Shadowing or reflecting the whole page** instead of part of it.
- **Losing a marker** under a thumb, under a reflection, or off the edge of the
  frame.
- **Cropping to "just the page"**, which removes the margin the detector needs.
- **Uploading HEIC** without checking that it converts.
