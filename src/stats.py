"""Lightweight statistical helpers for benchmark analysis."""

from __future__ import annotations

import random
from typing import Sequence


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Returns (mean, lower, upper). Empty input returns (nan, nan, nan) as 0.0 triple.
    """
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - ci
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    point = sum(values) / n
    return point, means[lo_idx], means[hi_idx]
