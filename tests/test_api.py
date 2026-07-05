#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Tests for the sitsfeats public API."""

import numpy as np
import pytest

import _sitsfeats
from sitsfeats import Features, feats, metrics
from sitsfeats.registry import MetricsRegistry, MetricTable

from .conftest import as_kernel_input
from .reference import REFERENCES


def test_feats_returns_features(matrix):
	"""feats() returns a Features with the requested columns, in order."""
	result = feats(data=as_kernel_input(matrix), metrics=["mean", "median", "iqr"])

	# test
	assert isinstance(result, Features)
	assert result.names == ["mean", "median", "iqr"]
	assert result.data.shape == (matrix.shape[0], 3)


def test_feats_column_order(matrix):
	"""Column j corresponds to names[j] regardless of request order."""
	# get the input matrix
	x = as_kernel_input(matrix)

	# compute the result
	result = feats(data=x, metrics=["iqr", "mean"])

	# test the first column
	np.testing.assert_allclose(result.data[:, 0], REFERENCES["iqr"](x))

	# test the second column
	np.testing.assert_allclose(result.data[:, 1], REFERENCES["mean"](x))


def test_features_helpers(matrix):
	"""`Features.to_dict()` to expose the data conveniently."""
	# get the input matrix
	x = as_kernel_input(matrix)

	# compute the result
	result = feats(data=x, metrics=["mean", "std"])

	# get the dictionary transformation
	as_dict = result.to_dict()

	# test the dictionary transformation
	assert set(as_dict) == {"mean", "std"}

	# test the mean column
	np.testing.assert_allclose(as_dict["mean"], REFERENCES["mean"](x))

	# test the array transformation
	np.testing.assert_array_equal(np.asarray(result), result.data)


def test_unknown_metric_raises(matrix):
	"""Requesting an unknown metric raises NotImplementedError."""
	with pytest.raises(NotImplementedError):
		feats(as_kernel_input(matrix), ["does_not_exist"])


def test_registry_reflects_extension():
	"""The Python registry mirrors exactly what the C++ extension declares."""
	# get the declared metrics
	declared = {name: group for name, group, _desc in _sitsfeats.list_metrics()}

	# test the registry names
	assert set(MetricsRegistry.names()) == set(declared)

	# test the group of the mean metric
	assert MetricsRegistry.group("mean") == "basic"

	# test the description of each metric
	# > every metric carries a non-empty, kernel-sourced description
	for _name, _group, desc in _sitsfeats.list_metrics():
		assert desc


def test_ndarray_input_returns_features(matrix):
	"""An ndarray dispatches to the array path and returns a Features."""
	assert isinstance(feats(data=as_kernel_input(matrix), metrics=["mean"]), Features)


def test_rejects_non_array_input():
	"""A non-array, non-xarray input is a TypeError, not a silent coercion."""
	with pytest.raises(TypeError):
		feats(data=[[1.0, 2.0, 3.0]], metrics=["mean"])


def test_ndarray_path_rejects_keywords(matrix):
	"""`dim` (and any kwarg) is meaningless for an ndarray and is rejected."""
	with pytest.raises(TypeError):
		feats(data=as_kernel_input(matrix), metrics=["mean"], dim="time")


def test_metrics_lists_all_names_sorted():
	"""metrics() returns a record per metric, sorted by name."""
	# get the metrics table
	table = metrics()

	# test the table
	assert isinstance(table, MetricTable)
	assert table.names == sorted(MetricsRegistry.names())
	assert "mean" in table.names and "area_ts" in table.names

	# test the records
	# > records carry the full (name, group, description) metadata
	by_name = {record.name: record for record in table}

	# test the mean record
	assert by_name["mean"].group == "basic"

	# test the description of each record
	assert all(record.description for record in table)


def test_metrics_filters_by_group():
	"""metrics(group=...) restricts the result to one family."""
	# get the polar metrics table
	polar = metrics(group="polar")

	# test the table
	assert polar.names == sorted(polar.names)
	assert all(record.group == "polar" for record in polar)

	# test the names
	assert "area_ts" in polar.names and "mean" not in polar.names


def test_metrics_output_feeds_feats(matrix):
	"""The names from metrics() are directly usable by feats()."""
	# get the basic metrics table
	basic = metrics(group="basic").names

	# compute the result
	result = feats(data=as_kernel_input(matrix), metrics=basic)

	# test the result
	assert result.names == basic


def test_metric_table_renders():
	"""MetricTable renders as aligned text and as an HTML table."""
	table = metrics(group="basic")

	# test the text representation
	text = repr(table)

	assert "name" in text and "group" in text and "description" in text
	assert "mean" in text

	# test the HTML representation
	html = table._repr_html_()
	assert html.startswith("<table>") and "<td" in html


def test_metric_table_exports():
	"""`to_dict()` and `to_pandas()` expose the table for downstream use."""
	table = metrics(group="polar")

	# test the dictionary transformation
	as_dicts = table.to_dict()
	assert {d["name"] for d in as_dicts} == set(table.names)

	# test the first dictionary
	assert set(as_dicts[0]) == {"name", "group", "description"}

	# test the pandas transformation
	pd = pytest.importorskip("pandas")
	frame = table.to_pandas()

	# test the pandas transformation
	assert isinstance(frame, pd.DataFrame)
	assert list(frame.columns) == ["name", "group", "description"]
	assert list(frame["name"]) == table.names
