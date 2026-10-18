"""Interquartile mean + percentile bootstrap confidence intervals, in the
spirit of Agarwal et al. (2021), "Deep RL at the Edge of the Statistical
Precipice" -- point estimates alone are misleading with the small seed
counts affordable here, and a wide, honest bootstrap CI is more informative
than a mean +/- std that implies more precision than a handful of seeds
actually supports.

Implemented directly with numpy/scipy rather than via the `rliable` package:
during this project, `rliable` 1.2.0's `library.get_interval_estimates`
turned out to be broken against `arch` 8.0.0 (needed for `arch` to import at
all under pandas 3.x) -- a real, version-specific incompatibility, not a
usage error. A ~20-line direct implementation of the same method avoids
depending on a package with that fragility, which also makes this notebook's
own dependency chain more reproducible for anyone re-running it later.
"""
from __future__ import annotations

import numpy as np
import scipy.stats


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean (25% trimmed mean). With very few seeds (<=3 per
    condition, as in most of this project's runs) this trims nothing and is
    equivalent to the plain mean -- noted here rather than left implicit,
    since the point of using this name is method-consistency with the wider
    literature, not that trimming is actually doing anything at this scale."""
    return float(scipy.stats.trim_mean(np.asarray(scores), proportiontocut=0.25))


def bootstrap_ci(scores: np.ndarray, reps: int = 50_000, ci: float = 95.0, seed: int = 0):
    """Percentile bootstrap CI for the IQM of `scores` (1D array, one value
    per seed). Returns (point_estimate, ci_lo, ci_hi)."""
    scores = np.asarray(scores)
    n = len(scores)
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(reps, n))
    boot_iqms = scipy.stats.trim_mean(scores[resample_indices], proportiontocut=0.25, axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot_iqms, [alpha, 100.0 - alpha])
    return iqm(scores), float(lo), float(hi)
