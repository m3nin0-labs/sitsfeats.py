#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Parity of the sitsfeats polar kernels with the Shapely/stmetrics reference.

This is the ground-truth check: it reconstructs the exact polygon-based
computation stmetrics performs with Shapely and compares it to our sitsfeats kernels.
"""

import numpy as np
import pytest

from .conftest import call_kernel

shapely = pytest.importorskip("shapely")

from shapely.geometry import LinearRing, Polygon, box  # noqa: E402

# List of all polar metrics
POLAR_METRICS = [
	"area_ts",
	"angle",
	"gyration_radius",
	"csi",
	"area_q1",
	"area_q2",
	"area_q3",
	"area_q4",
	"polar_balance",
]


def _polygon(ts):
	"""The stmetrics polar polygon for a time series."""
	r = np.abs(ts.astype(float))
	n = len(r)

	# create the ring
	ring = [
		[r[i] * np.cos(2 * np.pi * i / n), r[i] * np.sin(2 * np.pi * i / n)]
		for i in range(n)
	]

	# create the polygon
	return Polygon(LinearRing(ring)).buffer(0)


def _shapely_metrics(ts):
	"""All polar metrics computed the stmetrics way."""
	# create the polygon
	poly = _polygon(ts)
	r = np.abs(ts.astype(float))
	angles = np.linspace(0, 2 * np.pi, len(r))

	# get the coordinates and centroid
	coords = np.asarray(poly.exterior.coords)
	centroid = poly.centroid
	gyration = np.mean(np.hypot(coords[:, 0] - centroid.x, coords[:, 1] - centroid.y))

	# get the bounding box
	minx, miny, maxx, maxy = poly.bounds

	# create the quad boxes
	quad_boxes = [
		box(0, 0, maxx, maxy),
		box(minx, 0, 0, maxy),
		box(minx, miny, 0, 0),
		box(0, miny, maxx, 0),
	]

	# compute the quad areas
	quad_areas = [poly.intersection(b).area for b in quad_boxes]

	# return the metrics
	return {
		"area_ts": poly.area,
		"angle": angles[np.argmax(r)],
		"gyration_radius": gyration,
		"csi": poly.length**2 / (4 * np.pi * poly.area),
		"area_q1": quad_areas[0],
		"area_q2": quad_areas[1],
		"area_q3": quad_areas[2],
		"area_q4": quad_areas[3],
		"polar_balance": float(np.std(quad_areas)),
	}


@pytest.fixture
def series_matrix():
	"""Series matrix for parity checking."""
	# get the random number generator
	rng = np.random.default_rng(2024)

	# create the matrix
	return np.ascontiguousarray(rng.normal(5000.0, 800.0, size=(6, 29)))


@pytest.mark.parametrize("metric", POLAR_METRICS)
def test_polar_matches_stmetrics(metric, series_matrix):
	"""Each polar kernel reproduces the Shapely/stmetrics value."""
	# get the expected result
	expected = np.array([_shapely_metrics(row)[metric] for row in series_matrix])

	# compute the result
	result = call_kernel(metric, series_matrix)

	# test the result
	np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
