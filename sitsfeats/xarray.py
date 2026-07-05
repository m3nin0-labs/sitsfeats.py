#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""xarray integration for time-series feature extraction.

Internal implementation of the xarray path of `sitsfeats.feats`. Users do
not call into this module directly. Passing an `xarray` object to `sitsfeats.feats`
dispatches here. It extracts metrics from a labelled data cube while preserving
its spatial coordinates and attributes, delegating to the C++ kernels via
`xarray.apply_ufunc` with `dask="parallelized"` so chunked / out-of-core
cubes are handled transparently.

This module requires the optional `xarray` dependency::

    pip install "sitsfeats.py[xarray]"
"""

import numpy as np

try:
	import xarray as xr

except ImportError as exc:  # pragma: no cover - exercised only without xarray
	raise ImportError(
		"sitsfeats.xarray requires xarray. Install it with: "
		'pip install "sitsfeats.py[xarray]"'
	) from exc

import _sitsfeats


def _apply_block(block: np.ndarray, names: list[str]) -> np.ndarray:
	"""Compute metrics for one array block whose last axis is time.

	Args:
	    block (np.ndarray): Array of shape `(*spatial, n_time)`.

	    names (list[str]): Metric names.

	Returns:
	    np.ndarray: Array of shape `(*spatial, n_metrics)`.
	"""
	# convert to numpy array
	block = np.asarray(block, dtype=np.float64)

	# get the shape
	*spatial, n_time = block.shape

	# reshape to (n_pixels, n_time)
	flat = block.reshape(-1, n_time)  # (n_pixels, n_time)

	# compute metrics
	out = _sitsfeats.compute(names, flat)  # (n_pixels, n_metrics)

	# reshape to (*spatial, n_metrics)
	return out.reshape(*spatial, len(names))


def _extract_dataarray(da: xr.DataArray, metrics: list[str], dim: str) -> xr.Dataset:
	"""Extract metrics from a single DataArray cube into a Dataset.

	Args:
	    da (xr.DataArray): DataArray cube.

	    metrics (list[str]): Metric names.

	    dim (str): Name of the time dimension.

	Returns:
	    xr.Dataset: Dataset with one variable per metric.
	"""
	# check if the dimension is in the data array
	if dim not in da.dims:
		raise ValueError(f"Dimension {dim!r} not found in {da.dims}.")

	# apply the block function
	result = xr.apply_ufunc(
		_apply_block,
		da,
		input_core_dims=[[dim]],
		output_core_dims=[["metric"]],
		dask="parallelized",
		output_dtypes=[np.float64],
		dask_gufunc_kwargs={"output_sizes": {"metric": len(metrics)}},
		keep_attrs=True,
		kwargs={"names": metrics},
	)

	# assign the metric coordinates
	result = result.assign_coords(metric=metrics)

	# return!
	return result.to_dataset(dim="metric")


def run(
	obj: xr.DataArray | xr.Dataset,
	metrics: list[str],
	dim: str = "time",
) -> xr.Dataset:
	"""Extract time-series metrics from a labelled data cube.

	Internal entry point for the `xarray` path of `sitsfeats.feats`. The
	metric list is already validated by the caller.

	Args:
	    obj (xr.DataArray | xr.Dataset): Input cube. Must contain the time
	                                     dimension `dim`. For a Dataset, every
	                                     variable carrying `dim` is processed.

	    metrics (list[str]): Metric names.

	    dim (str): Name of the time dimension. Defaults to `"time"`.

	Returns:
	    xr.Dataset: One variable per metric (named `"<metric>"` for a DataArray
	                input, or `<variable>_<metric>` for a Dataset input), with
	                the cube spatial coordinates and attributes preserved.
	"""
	# check if the object is a data array
	if isinstance(obj, xr.DataArray):
		# if it is a data array, extract the metrics directly
		return _extract_dataarray(obj, metrics, dim)

	# in the case of a dataset, extract the metrics per variable
	if isinstance(obj, xr.Dataset):
		# extract the metrics per variable
		per_var = []

		for name, da in obj.data_vars.items():
			# check if the dimension is in the data array
			if dim not in da.dims:
				continue

			# if so, extract the metrics
			ds = _extract_dataarray(da, metrics, dim)

			# rename the metrics
			per_var.append(ds.rename({m: f"{name}_{m}" for m in metrics}))

		# check if no variables were extracted
		if not per_var:
			raise ValueError(f"No variable in the Dataset has dimension {dim!r}.")

		# merge the variables
		return xr.merge(per_var)

	# fallback: raise error in caso of unexpected input
	raise TypeError(
		f"Expected an xarray DataArray or Dataset, got {type(obj).__name__}."
	)
