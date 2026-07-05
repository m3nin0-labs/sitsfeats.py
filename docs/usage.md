# Usage

The `sitsfeats.py` provides the main functions to select metrics and calculate them: `feats` and `metrics`. The usage of these functions are presented in the following sections.

## The `feats` function

`feats` is the single entry point. It dispatches on the type of its **first** argument:

- a **numpy array** (each row a time series) → a `Features` object;
- an **xarray** `DataArray`/`Dataset` cube → an `xarray.Dataset` (see [Data cubes](xarray.md)).

The example below shows the generation of a dataset and application of feats

```python
import numpy as np
from sitsfeats import feats

# 5 time series of 36 observations each.
data = np.random.default_rng(0).normal(5000, 800, size=(5, 36))

# calculate feats
result = feats(data, ["median", "skew", "iqr"])
```

The result holds a stacked `(n_series, n_metrics)` array plus the column names:

```python
result.names        # metrics, e.g., ['median', 'skew', 'iqr']
result.data         # ndarray of shape (5, 3); column j corresponds to names[j]
result.data[:, 0]   # the 'median' column
```

Helpers:

To support the usage of the `feats` function, there are also some auxiliary functions.

```python
result.to_dict()    # {'median': array(...), 'skew': array(...), 'iqr': array(...)}
np.asarray(result)  # the stacked (5, 3) matrix
```

Any memory layout (C- or Fortran-contiguous) and `float`/`int` input is accepted. The input array is never modified.

## Discovering metrics

Use `metrics()` to see everything available, with each metric's family and a short description (the data comes straight from the compiled kernels, so it never drifts). It renders as a table in the terminal and in Jupyter:

```python
from sitsfeats import metrics

metrics()               # all metrics
metrics(group="polar")  # just one family
```

```text
name             group  description
abs_sum          basic  Sum of absolute values
amd              basic  Mean absolute first difference
amplitude        basic  Range (max - min)
...
area_ts          polar  Area of the polar-plot polygon
polar_balance    polar  Std-dev of the four quadrant areas
```

The result behaves as a sequence of `(name, group, description)` records. To get the bare names to feed `feats`, use `.names`. To export, use `.to_dict()` or `.to_pandas()` (pandas is optional and imported only on demand):

```python
metrics(group="polar").names                # ['angle', 'area_q1', ...]
feats(data, metrics(group="basic").names)   # compute a whole family
metrics().to_pandas()                       # a DataFrame, if you have pandas
```

See the [Metrics catalogue](metrics.md) for the full definition of each one.

## Mixing families

Basic and polar metrics can be requested together in a single call. They are computed in one pass:

```python
feats(data, ["mean", "amplitude", "area_ts", "polar_balance"])
```

## Errors

Requesting an unknown metric raises `NotImplementedError`:

```python
feats(data, ["not_a_metric"])   # NotImplementedError
```
