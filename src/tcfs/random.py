"""Utilities for repeatable pseudo-random behavior."""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Seed Python and NumPy, and return a modern NumPy generator.

    ``PYTHONHASHSEED`` applies to child processes. For hash randomization in the
    current interpreter, launch Python with the same environment value already set.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

