import string
from fractions import Fraction
from math import sqrt

import numpy as np

from wlstar.base.semiring import (
    Boolean,
    Count,
    Integer,
    MaxPlus,
    Rational,
    Real,
    String,
    Tropical,
)


def _random_weight(semiring, rng=None, **kwargs):  # noqa: C901
    if rng is None:
        rng = np.random.default_rng()
    if semiring is String:
        str_len = int(rng.random() * 8 + 1)
        return semiring(
            "".join(rng.choice(string.ascii_lowercase) for _ in range(str_len))
        )

    elif semiring is Boolean:
        return semiring(True)

    elif semiring is Real:
        tol = 1e-3
        s = kwargs.get("divide_by", 2)
        random_weight = round(rng.random() / s, 3)
        while random_weight < sqrt(tol):
            random_weight = round(rng.random() / s, 3)
        return semiring(random_weight)

    elif semiring is Rational:
        # return semiring(Fraction(f"{rng.integers(1, 100)}/{rng.integers(1, 2)}"))
        return semiring(Fraction(f"{rng.integers(1, 10)}/1"))

    elif semiring is Tropical:
        return semiring(rng.integers(0, 50))

    elif semiring is Integer:
        return semiring(rng.integers(1, 10))

    elif semiring is MaxPlus:
        return semiring(rng.integers(-10, -1))

    elif semiring is Count:
        return semiring(1.0)
