#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Property-based tests: invariants that must hold for any valid input."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from .conftest import as_kernel_input, call_kernel

# Bounded, finite, well-scaled values keep FFT-based metrics numerically stable.
_elements = st.floats(
	min_value=-1.0e3,
	max_value=1.0e3,
	allow_nan=False,
	allow_infinity=False,
)


def _matrices():
	"""Strategy for matrices with at least 2 time steps."""
	# create the shape
	shape = st.tuples(
		st.integers(min_value=1, max_value=6),
		st.integers(min_value=2, max_value=24),
	)

	# create the matrix
	return arrays(
		dtype=np.float64,
		shape=shape,
		elements=_elements,
	)


@settings(max_examples=200, deadline=None)
@given(_matrices())
def test_shapes(x):
	"""Every reduction returns exactly one value per row."""
	# get the number of series
	n = x.shape[0]

	# convert to the kernel input
	xf = as_kernel_input(x)

	for metric in ("mean", "std", "median", "amplitude", "iqr", "mse", "amd"):
		# compute the result
		assert call_kernel(metric, xf).shape == (n,)


@settings(max_examples=200, deadline=None)
@given(_matrices())
def test_ordering_invariants(x):
	"""min <= {mean, median, quantiles} <= max, and quartiles are ordered."""
	# get the input matrix
	xf = as_kernel_input(x)

	# get the minimum and maximum
	lo = call_kernel("min", xf)
	hi = call_kernel("max", xf)

	# get the tolerance
	tol = 1e-6 * (1.0 + np.abs(hi))

	# test the ordering invariants
	for metric in ("mean", "median", "fqr", "tqr"):
		# compute the result
		v = call_kernel(metric, xf)

		# test the ordering
		assert np.all(v >= lo - tol)
		assert np.all(v <= hi + tol)

	# test the ordering of the quartiles
	fqr, med, tqr = (call_kernel(m, xf) for m in ("fqr", "median", "tqr"))

	assert np.all(fqr <= med + tol)
	assert np.all(med <= tqr + tol)


@settings(max_examples=200, deadline=None)
@given(_matrices())
def test_algebraic_identities(x):
	"""amplitude == max-min, iqr == tqr-fqr, and non-negativity of spreads."""
	# convert to the kernel input
	xf = as_kernel_input(x)

	# test the algebraic identities
	amp = call_kernel("amplitude", xf)

	np.testing.assert_allclose(
		amp, call_kernel("max", xf) - call_kernel("min", xf), rtol=1e-9, atol=1e-9
	)

	# test the algebraic identity for the interquartile range
	np.testing.assert_allclose(
		call_kernel("iqr", xf),
		call_kernel("tqr", xf) - call_kernel("fqr", xf),
		rtol=1e-9,
		atol=1e-9,
	)

	# test the algebraic identity for the mean absolute first difference
	assert np.all(call_kernel("amd", xf) <= call_kernel("fslope", xf) + 1e-6)

	# test the non-negativity of the spreads
	for metric in ("amplitude", "iqr", "std", "amd", "fslope", "abs_sum"):
		# compute the result
		assert np.all(call_kernel(metric, xf) >= -1e-9)


@settings(max_examples=50, deadline=None)
@given(
	st.integers(min_value=1, max_value=5),
	st.integers(min_value=2, max_value=20),
	_elements,
)
def test_constant_rows(n_series, n_steps, value):
	"""Constant time series have zero spread and mean/median equal to the value."""
	# create the matrix
	x = as_kernel_input(
		np.full(
			(n_series, n_steps),
			value,
		)
	)

	# test the algebraic identities
	np.testing.assert_allclose(call_kernel("amplitude", x), 0.0, atol=1e-9)
	np.testing.assert_allclose(call_kernel("std", x), 0.0, atol=1e-9)
	np.testing.assert_allclose(call_kernel("iqr", x), 0.0, atol=1e-9)

	# test the mean and median
	np.testing.assert_allclose(call_kernel("mean", x), value, rtol=1e-9, atol=1e-9)
	np.testing.assert_allclose(call_kernel("median", x), value, rtol=1e-9, atol=1e-9)
