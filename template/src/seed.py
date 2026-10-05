"""Make a run reproducible.

Call this once at the start of every script. An experiment you cannot repeat is
an anecdote, and you will want to compare runs all semester.

Seed every generator you use: Python's `random`, NumPy, and PyTorch on the CPU
and on every GPU. Some GPU kernels stay non-deterministic unless you also set
`torch.use_deterministic_algorithms(True)`, which costs speed.
"""


def set_seed(seed: int) -> None:
    raise NotImplementedError
