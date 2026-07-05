#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Tests for the optional xarray integration."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from sitsfeats import feats  # noqa: E402

#
# Test metrics
#
METRICS = ["mean", "skew", "iqr"]


@pytest.fixture
def cube():
	"""A synthetic (time, y, x) data cube with coordinates and attributes."""
	nt, ny, nx = 24, 4, 5
	rng = np.random.default_rng(42)

	# create the data
	data = rng.normal(
		loc=5000.0,
		scale=800.0,
		size=(nt, ny, nx),
	)

	# create the data array
	return xr.DataArray(
		data,
		dims=("time", "y", "x"),
		coords={
			"time": np.arange(nt),
			"y": np.linspace(0, 1, ny),
			"x": np.linspace(0, 1, nx),
		},
		attrs={"sensor": "test"},
		name="ndvi",
	)


def _reference(cube):
	"""feats() applied to the manually flattened cube -> {metric: (y, x)}."""
	# get the size of the cube
	ny, nx = cube.sizes["y"], cube.sizes["x"]

	# flatten the cube
	flat = cube.transpose("y", "x", "time").values.reshape(ny * nx, -1)

	# compute the result
	result = feats(flat, METRICS)

	# return the result
	return {m: result.to_dict()[m].reshape(ny, nx) for m in METRICS}


def test_returns_dataset_with_one_var_per_metric(cube):
	"""feats() returns a Dataset with one variable per metric."""
	# get the result
	out = feats(cube, METRICS)

	# tests
	assert isinstance(out, xr.Dataset)
	assert set(out.data_vars) == set(METRICS)

	for m in METRICS:
		assert out[m].dims == ("y", "x")


def test_values_match_feats(cube):
	"""The values from feats() match the manually flattened cube."""
	out = feats(cube, METRICS)
	ref = _reference(cube)

	# test the values
	for m in METRICS:
		np.testing.assert_allclose(out[m].values, ref[m], rtol=1e-9, atol=1e-9)


def test_coords_and_attrs_preserved(cube):
	"""The coordinates and attributes are preserved."""
	out = feats(cube, METRICS)

	# test the coordinates
	np.testing.assert_array_equal(out["y"].values, cube["y"].values)
	np.testing.assert_array_equal(out["x"].values, cube["x"].values)

	# test the attributes
	assert "time" not in out.dims
	assert out.attrs.get("sensor") == "test"


def test_dask_parallelized_matches_numpy(cube):
	"""The chunked (Dask) path agrees with the eager path."""
	pytest.importorskip("dask")

	# create the chunked cube
	chunked = cube.chunk({"y": 2, "x": 2})  # time stays a single chunk

	# compute the result
	out = feats(chunked, METRICS)

	# test that the result is lazy
	assert out["mean"].chunks is not None

	# compute the result
	ref = _reference(cube)

	# test the result
	computed = out.compute()

	for m in METRICS:
		np.testing.assert_allclose(computed[m].values, ref[m], rtol=1e-9, atol=1e-9)


def test_custom_time_dim_name():
	"""The time dimension can be named anything."""
	# get the random number generator
	rng = np.random.default_rng(0)

	# create the data array
	da = xr.DataArray(
		data=rng.normal(size=(10, 3)),
		dims=("date", "site"),
		name="band",
	)

	# compute the result
	out = feats(data=da, metrics=["mean"], dim="date")

	# tests
	assert out["mean"].dims == ("site",)


def test_missing_dim_raises(cube):
	"""A missing time dimension raises a ValueError."""
	with pytest.raises(ValueError, match="not found"):
		feats(data=cube, metrics=METRICS, dim="nope")


def test_unknown_metric_raises(cube):
	"""An unknown metric raises a NotImplementedError."""
	with pytest.raises(NotImplementedError):
		feats(data=cube, metrics=["does_not_exist"])


def test_dataset_input_prefixes_variables(cube):
	"""A Dataset input prefixes the variables."""
	ds = xr.Dataset({"ndvi": cube, "evi": cube + 100.0})

	out = feats(data=ds, metrics=["mean"])

	# test the result
	assert set(out.data_vars) == {"ndvi_mean", "evi_mean"}

	# compute the reference
	ref = _reference(cube)

	# test the result
	np.testing.assert_allclose(out["ndvi_mean"].values, ref["mean"], rtol=1e-9)
