#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Correctness of the compiled sitsfeats kernels against numpy/scipy references."""

import numpy as np
import pytest

from .conftest import as_kernel_input, call_kernel
from .reference import REFERENCES

#
# List of all metrics
#
ALL_METRICS = sorted(REFERENCES)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_kernel_matches_reference(metric, matrix):
	"""Each sitsfeats kernel matches its independent numpy/scipy reference."""
	# get the expected result
	expected = REFERENCES[metric](matrix)

	# compute the result
	result = call_kernel(metric, as_kernel_input(matrix))

	# test the result
	assert result.shape == expected.shape

	np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)
