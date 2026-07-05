#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Shared test fixtures and helpers."""

import numpy as np
import pytest

import _sitsfeats


def call_kernel(name, x):
	"""Call a raw C++ kernel by base name and return a flat 1-D result.

	Args:
	    name (str): Metric base name (e.g. ``"mean"`` -> ``C_ts_mean``).

	    x (np.ndarray): Input matrix (rows = series, cols = time steps).

	Returns:
	    np.ndarray: 1-D result, one value per row.
	"""
	# get the function
	fnc = getattr(_sitsfeats, f"C_ts_{name}")

	# call function selected
	return np.asarray(fnc(x)).ravel()


def as_kernel_input(x):
	"""Coerce data to a `float64` array for the kernels.

	The `nanobind` + Eigen `bridge` accepts any memory layout.
	This helper only pins the dtype so reference comparisons are exact.
	"""
	return np.ascontiguousarray(x, dtype=np.float64)


@pytest.fixture
def rng():
	"""Deterministic random generator."""
	return np.random.default_rng(20260704)


@pytest.fixture
def matrix(rng):
	"""A representative (n_series, n_steps) time-series matrix."""
	return rng.normal(loc=5000.0, scale=800.0, size=(7, 53))
