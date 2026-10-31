# data

Nothing in here is tracked by git. The directory layout is, so that everyone's
paths match.

```
raw/train.csv                   the labels and the grouping columns
raw/train/                      1841 training photographs (.jpg)
raw/test/                       1628 test photographs, from the competition's Data tab
raw/sample_submission.csv       from the competition's Data tab
work/reference/                 resized copies, written by python -m src.prepare
```

Treat `raw/` as read only. If a script needs a different form of the data it
writes it to `work/`, so that deleting `work/` and regenerating it is always
safe.
