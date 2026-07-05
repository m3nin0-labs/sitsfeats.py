#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Memory-layout contract for the sitsfeats kernels.

With the nanobind + Eigen bridge the kernels accept any NumPy layout via
strides, compute the same correct result, and never mutate the caller's array.
These tests pin that contract.
"""

import numpy as np
import pytest

from .conftest import call_kernel
from .reference import ref_mean


@pytest.fixture
def base_matrix():
	"""Base matrix. What `np.genfromtxt` or users typically produce."""
	return np.ascontiguousarray(
		np.random.default_rng(7).normal(size=(6, 31)), dtype=np.float64
	)


@pytest.mark.parametrize("order", ["C", "F"])
def test_layout_independent_result(base_matrix, order):
	"""C- and F-contiguous input give the same correct result."""
	# get the input matrix
	x = np.asarray(base_matrix, order=order)

	# compute the result
	result = call_kernel("mean", x)

	# test
	np.testing.assert_allclose(result, ref_mean(base_matrix), rtol=1e-9)


@pytest.mark.parametrize("order", ["C", "F"])
def test_input_not_mutated(base_matrix, order):
	"""Calling a sitsfeats kernel must never modify the input array."""
	# get the input matrix
	x = np.asarray(base_matrix, order=order)

	# get a copy of the input matrix
	before = x.copy()

	# call the kernel for each metric
	for metric in ("mean", "std", "skew", "max", "iqr"):
		call_kernel(metric, x)

	# test that the input is unchanged
	np.testing.assert_array_equal(x, before)
