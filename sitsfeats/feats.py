#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""feats module."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

import _sitsfeats
from sitsfeats.registry import MetricsRegistry

if TYPE_CHECKING:
	import xarray as xr


class Features(NamedTuple):
	"""Result of a feature extraction over a plain array.

	Attributes:
	    data (np.ndarray): Stacked metrics with shape ``(n_series, n_metrics)``.
	                       Column ``j`` holds the metric named ``names[j]``.

	    names (list[str]): Metric names, in column order.
	"""

	data: np.ndarray
	"""Stacked metrics with shape ``(n_series, n_metrics)``."""

	names: list[str]
	"""Metric names, in column order."""

	def to_dict(self) -> dict[str, np.ndarray]:
		"""Return the metrics as a ``{name: column}`` mapping.

		Returns:
		    dict[str, np.ndarray]: A dictionary with metric names as keys and
		                                the corresponding columns as values.
		"""
		return {name: self.data[:, i] for i, name in enumerate(self.names)}

	def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
		"""Expose the stacked matrix to `np.asarray`.

		Args:
		    dtype (dtype | None): The dtype of the returned array.

		Returns:
		    np.ndarray: The stacked matrix as a numpy array.
		"""
		return np.asarray(self.data, dtype=dtype)


def _check_metrics(metrics: list[str]) -> None:
	"""Check that the given metrics are available.

	Args:
	    metrics (list[str]): Metric names.

	Raises:
	    NotImplementedError: If a metric is not available.
	"""
	for metric in metrics:
		MetricsRegistry.exists(metric, raise_error=True)


@singledispatch
def _dispatch(
	data: np.ndarray | xr.DataArray | xr.Dataset, metrics: list[str], **kwargs: str
) -> Features | xr.Dataset:
	"""Dispatch on the input type. The default handles xarray.

	Args:
	    data (np.ndarray | xr.DataArray | xr.Dataset): Time-series data - a 2-D array
	                                        (rows are series) or an xarray cube.

	    metrics (list[str]): Metric names to extract.

	    **kwargs: Forwarded to the input-specific handler (e.g. `dim` for an
	              xarray cube). Passing keywords for a numpy array input is an error.

	Returns:
	    Features | xr.Dataset: A `Features` object for a numpy array input, or an
	                            `xarray.Dataset` for an `xarray` object.

	Raises:
	    TypeError: If `data` is not a numpy array or an xarray object.
	"""
	# check if data is an xarray object
	if type(data).__module__.split(".", 1)[0] == "xarray":
		# lazy on purpose: keeps the optional xarray dependency out of
		# `import sitsfeats`. It is only reached once an xarray object arrives.
		from sitsfeats import xarray as _xarray  # noqa: PLC0415

		# run xarray
		return _xarray.run(cast("xr.DataArray | xr.Dataset", data), metrics, **kwargs)

	# raise error
	raise TypeError(
		"feats() accepts a numpy.ndarray or an xarray DataArray/Dataset, "
		f"got {type(data).__name__!r}."
	)


@_dispatch.register
def _(data: np.ndarray, metrics: list[str], **kwargs) -> Features:
	"""Handle a numpy array input.

	Args:
	    data (np.ndarray): Time-series data - a 2-D array (rows are series).

	    metrics (list[str]): Metric names to extract.

	    **kwargs: Forwarded to the input-specific handler
	            (e.g. `dim` for an `xarray` cube). Passing keywords for a
	            `numpy.ndarray` input is meaningless and raises an error.

	Returns:
	    Features: A `Features` object with the stacked metrics and the metric names.

	Raises:
	    TypeError: If `kwargs` are passed for a `numpy.ndarray` input.
	"""
	# kwargs are not allowed for ndarray
	if kwargs:
		raise TypeError(
			"feats() got unexpected keyword argument(s) for ndarray input: "
			f"{', '.join(sorted(kwargs))}."
		)

	# compute metrics
	matrix = _sitsfeats.compute(metrics, data)

	# return!
	return Features(data=matrix, names=metrics)


def feats(
	data: np.ndarray | xr.DataArray | xr.Dataset,
	metrics: list[str],
	**kwargs: str,
) -> Features | xr.Dataset:
	"""Extract time-series metrics (features) from satellite image data.

	A single entry point that dispatches on the type of `data`:

	- `numpy.ndarray` - each row is a time series. Returns a `Features`
	  with a stacked `(n_series, n_metrics)` array and the column names.

	- `xarray.DataArray / Dataset` - a labelled data cube. Returns an
	  `xarray.Dataset` with one variable per metric, preserving spatial
	  coordinates and attributes (chunked/Dask cubes are processed lazily). This
	  path requires the optional `xarray` dependency and accepts a `dim`
	  keyword naming the time dimension (default `"time"`).

	Args:
	    data (np.ndarray | xr.DataArray | xr.Dataset): Time-series data - a 2-D array
	                                    (rows are series) or an xarray cube.

	    metrics (list[str]): Metric names.

	    **kwargs: Forwarded to the input-specific handler
	            (e.g. `dim` for an `xarray` cube). Passing keywords for a
	            `numpy.ndarray` input is meaningless and raises an error.

	Returns:
	    Features | xarray.Dataset: A `Features` for an array input, or an
	    `xarray.Dataset` for an `xarray` cube.

	Raises:
	    NotImplementedError: If a metric is not available.
	    TypeError: If `data` is not a `numpy.ndarray` or an `xarray` object.
	"""
	# get list of supported metrics
	metrics = list(metrics)

	# check if metrics are supported
	_check_metrics(metrics)

	# dispatch to the appropriate handler
	return _dispatch(data, metrics, **kwargs)
