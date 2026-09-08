"""Make a run reproducible.

Call this once at the start of every script. An experiment you cannot repeat is
an anecdote, and you will want to compare runs all semester.
"""


def set_seed(seed: int) -> None:
    raise NotImplementedError
