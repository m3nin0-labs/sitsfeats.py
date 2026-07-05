# %%
#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# %% [markdown]
# # Extracting features from an xarray data cube
#
# The same `feats` entry point also accepts a labelled `xarray` cube, a `(time, y, x)` stack of satellite observations, and returns a `Dataset` with one variable per metric, keeping the spatial coordinates and attributes. This path needs the optional dependency:
#
# ```shell
# pip install "sitsfeats.py[xarray]"
# ```
#
# Run it with `uv run --extra xarray python examples/ts-xarray.py`, or open it as a notebook (the `# %%` cells load directly in Jupyter and VS Code).

# %%
import numpy as np
import xarray as xr

from sitsfeats import feats

# %% [markdown]
# ## 1) Build a small synthetic cube
#
# A `(time, y, x)` cube with 24 dates over a 4x5 grid, carrying coordinates and a
# `sensor` attribute so we can see them survive the extraction.

# %%
# random generator
rng = np.random.default_rng(42)

# cube properties
nt, ny, nx = 24, 4, 5

# define cube
cube = xr.DataArray(
	rng.normal(loc=5000.0, scale=800.0, size=(nt, ny, nx)),
	dims=("time", "y", "x"),
	coords={
		"time": np.arange(nt),
		"y": np.linspace(0, 1, ny),
		"x": np.linspace(0, 1, nx),
	},
	attrs={"sensor": "example"},
	name="ndvi",
)

# show cube
cube

# %% [markdown]
# ## 2) Extract metrics over the time dimension
#
# `feats` reduces along `dim` (default `"time"`) and returns a `Dataset` with one
# variable per metric, each shaped like the spatial grid `(y, x)`.

# %%
features = feats(cube, ["mean", "skew", "iqr"], dim="time")

features

# %% [markdown]
# ## 3) Coordinates and attributes are preserved
#
# The time dimension is gone, but the spatial coordinates and the cube attributes
# carry through, and each metric is a named variable you can select directly.

# %%
features["skew"]
