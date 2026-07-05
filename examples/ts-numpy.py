# %%
#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# %% [markdown]
# # Extracting features from numpy time series
#
# `sitsfeats` turns each row of a 2-D array (one satellite image time series per row) into a handful of summary *metrics*, the mean, the interquartile range, the area of its polar plot, and so on. This walkthrough loads a small example dataset and extracts a few of those metrics step by step.
#
# Run it as a script with `uv run python examples/ts-numpy.py`, or open it as a notebook - Jupyter and VS Code read the `# %%` cells directly.

# %%

import numpy as np

from sitsfeats import feats, metrics

# %% [markdown]
# ## 1) Load the example series
#
# The CSV ships one time series per row (24 observations each).

# %%
data_path = "data.csv"
data = np.genfromtxt(data_path, delimiter=",", skip_header=1)

data.shape  # (n_series, n_steps)

# %% [markdown]
# ## 2) Extract a couple of basic metrics
#
# `feats` takes the data and a list of metric names. It returns a `Features`
# object holding a stacked `(n_series, n_metrics)` matrix plus the column names,
# so column `j` always corresponds to `result.names[j]`.

# %%
result = feats(data, ["median", "iqr", "skew"])

result.names
# > ['median', 'iqr', 'skew']

# %%
result.data
# > one row per series, one column per requested metric

# %% [markdown]
# ## 3) Get a name to column mapping
#
# When you want the columns by name rather than by position, `to_dict()` returns
# a `{metric: column}` mapping. A `Features` also behaves like an array, so
# `np.asarray(result)` yields the stacked matrix directly.

# %%
result.to_dict()["skew"]

# %% [markdown]
# ## 4) Discover the available metrics
#
# `metrics()` lists metric with its family and description. It renders as an aligned table in the terminal and as an HTML table in a notebook. Pass `group=...` to focus on one family.

# %%
metrics()

# %%
metrics(group="polar")

# %% [markdown]
# ## 5) Extract a polar metric
#
# The polar family maps each series to a closed polygon (the Körting polar plot)
# and measures its geometry. The names from `metrics()` feed straight back into
# `feats`, so any metric is one call away.

# %%
polar = feats(data, ["area_ts", "polar_balance"])

polar.to_dict()
