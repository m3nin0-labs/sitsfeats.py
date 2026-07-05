#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Pure numpy/scipy reference implementations of the basic-family kernels.

These are written independently of the compiled extension so the test suite can
assert what each basic metric *should* compute. The polar family has its own
ground truth (Shapely/stmetrics) in `test_polar.py`. Conventions are pinned to
match the C++ kernels (verified empirically):

* `std`   -> sample standard deviation (`ddof=1`).

* `skew`  -> adjusted Fisher-Pearson coefficient (`scipy` `bias=False`).

* `kurt`  -> Pearson kurtosis, non-excess, biased (normal == 3.0).

* `fqr` / `tqr` -> the **Hazen** quantile convention (`numpy` `method="hazen"`),
                  preserved from the original Armadillo implementation.

* `mse`   -> mean power spectrum, `mean(|fft|**2)` over frequency bins.

Every metric reduces along `axis=1` (rows = time series, cols = time steps) and
returns a 1-D array of length `n_series`.
"""

import numpy as np
from scipy import stats

#: Quantile convention used by Armadillo's ``quantile``.
_QUANTILE_METHOD = "hazen"


def ref_max(x):
	return np.max(x, axis=1)


def ref_min(x):
	return np.min(x, axis=1)


def ref_mean(x):
	return np.mean(x, axis=1)


def ref_median(x):
	return np.median(x, axis=1)


def ref_sum(x):
	return np.sum(x, axis=1)


def ref_std(x):
	return np.std(x, axis=1, ddof=1)


def ref_skew(x):
	# Adjusted Fisher-Pearson standardized moment coefficient (G1).
	return stats.skew(x, axis=1, bias=False)


def ref_kurt(x):
	# Pearson kurtosis (non-excess, biased): normal distribution -> 3.0.
	return stats.kurtosis(x, axis=1, fisher=False, bias=True)


def ref_amplitude(x):
	return np.max(x, axis=1) - np.min(x, axis=1)


def ref_fslope(x):
	return np.max(np.abs(np.diff(x, axis=1)), axis=1)


def ref_abs_sum(x):
	return np.sum(np.abs(x), axis=1)


def ref_amd(x):
	return np.mean(np.abs(np.diff(x, axis=1)), axis=1)


def ref_mse(x):
	return np.mean(np.abs(np.fft.fft(x, axis=1)) ** 2, axis=1)


def ref_fqr(x):
	return np.quantile(x, 0.25, axis=1, method=_QUANTILE_METHOD)


def ref_tqr(x):
	return np.quantile(x, 0.75, axis=1, method=_QUANTILE_METHOD)


def ref_iqr(x):
	return ref_tqr(x) - ref_fqr(x)


# Basic-family metric names mapped to
# their reference implementation
REFERENCES = {
	"max": ref_max,
	"min": ref_min,
	"mean": ref_mean,
	"median": ref_median,
	"sum": ref_sum,
	"std": ref_std,
	"skew": ref_skew,
	"kurt": ref_kurt,
	"amplitude": ref_amplitude,
	"fslope": ref_fslope,
	"abs_sum": ref_abs_sum,
	"amd": ref_amd,
	"mse": ref_mse,
	"fqr": ref_fqr,
	"tqr": ref_tqr,
	"iqr": ref_iqr,
}
