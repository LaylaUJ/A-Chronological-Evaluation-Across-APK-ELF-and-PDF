import random

import numpy as np
import pytest

from tcfs.random import seed_everything


def test_seed_everything_is_repeatable() -> None:
    first_generator = seed_everything(123)
    first_values = (random.random(), np.random.random(), first_generator.random())

    second_generator = seed_everything(123)
    second_values = (random.random(), np.random.random(), second_generator.random())

    assert first_values == second_values


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_seed_everything_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        seed_everything(seed)  # type: ignore[arg-type]

