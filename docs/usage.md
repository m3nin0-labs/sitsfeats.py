# Usage

`sitsfeats.py` gives you two functions: `feats`, which computes metrics, and `metrics`, which tells you what are the metrics available. This page walks through both functions.

## The `feats` function

`feats` is the single entry point of the package. It dispatches on the type of its first argument. Give it a numpy array, where each row is a time series, and it returns a `Features` object. Give it an `xarray.DataArray` or `xarray.Dataset` and it returns an `xarray.Dataset`, as shown below.

Start with five time series of 36 observations each:

```python
import numpy as np
from sitsfeats import feats

data = np.random.default_rng(0).normal(5000, 800, size=(5, 36))
```

Ask for the metrics you want, by name:

```python
result = feats(data, ["median", "skew", "iqr"])
```

The result holds the metric names and a stacked `(n_series, n_metrics)` array:

```python
result.names
# ['median', 'skew', 'iqr']

result.data.shape
# (5, 3)
```

Column `j` corresponds to `result.names[j]`, so the medians of the five series are the first column:

```python
result.data[:, 0]
# array([4895.7442, 5301.001 , 4953.904 , 5243.2148, 4914.0052])
```

When you would rather address the columns by name than by position, `to_dict()` returns a `{metric: column}` mapping, in the order you requested the metrics:

```python
result.to_dict()
# {'median': array([4895.7442, 5301.001 , 4953.904 , 5243.2148, 4914.0052]),
#  'skew':   array([-0.3461, -0.3394,  0.2875, -0.8561, -0.262 ]),
#  'iqr':    array([ 802.1748, 1427.1243, 1302.6574, 1112.6045,  887.0957])}
```

Which makes a single metric a lookup away:

```python
result.to_dict()["skew"]
# array([-0.3461, -0.3394,  0.2875, -0.8561, -0.262 ])
```

And a `Features` also behaves like an array, so `np.asarray` hands you the stacked matrix directly:

```python
np.asarray(result).shape
# (5, 3)
```

## Working with data cubes

The same `feats` entry point also accepts a labelled `xarray` cube, preserving its spatial coordinates and attributes. This path needs the optional extra:

```bash
pip install "sitsfeats[xarray]"
```

Take a cube with dimensions `(time, y, x)` and pass it as the first argument. The `dim` names the dimension to reduce over, and defaults to `"time"`:

```python
import xarray as xr
from sitsfeats import feats

# cube: an xarray.DataArray with dims (time, y, x)
features = feats(cube, ["mean", "skew", "iqr"], dim="time")
```

The result is an `xarray.Dataset` with one variable per metric:

```python
features
# Dataset with data vars: mean, skew, iqr
```

The time dimension is gone, so each metric is shaped like the cube's spatial grid:

```python
features["skew"].dims
# ('y', 'x')
```

The spatial coordinates and the cube attributes carry through untouched:

```python
features["y"], features["x"]   # original coordinates, preserved
features.attrs                 # original attributes, preserved
```

### Chunked cubes with Dask

The cube path is built on `xarray.apply_ufunc` with `dask="parallelized"`. If the cube is chunked, the computation stays lazy and runs per chunk, in parallel and without loading the whole cube into memory:

```python
chunked = cube.chunk({"y": 256, "x": 256})
features = feats(chunked, ["mean", "amplitude"])
```

Nothing has been computed at this point:

```python
features["mean"].data
# a dask array
```

Ask for the values when you want them:

```python
result = features.compute()
```

!!! note
    The time dimension must not be chunked, since it is the reduction axis. Chunk the
    spatial dimensions only.

### Datasets

A `Dataset` input is processed one variable at a time, and the output variables are named `<variable>_<metric>`:

```python
ds = xr.Dataset({"ndvi": ndvi_cube, "evi": evi_cube})

feats(ds, ["mean"])
# data vars: ndvi_mean, evi_mean
```

## Discovering metrics

You can use `metrics()` to see metrics available in the package. Each entry carries its family and a short description:

```python
from sitsfeats import metrics

metrics()
```

It renders as an aligned table in the terminal and as an HTML table in Jupyter:

```text
name             group  description
abs_sum          basic  Sum of absolute values
amd              basic  Mean absolute first difference
amplitude        basic  Range (max - min)
...
area_ts          polar  Area of the polar-plot polygon
polar_balance    polar  Std-dev of the four quadrant areas
```

Pass `group` to focus on a single family:

```python
metrics(group="polar")
```

The result behaves as a sequence of `(name, group, description)` records. To get the bare names, the ones `feats` expects, use `.names`:

```python
metrics(group="polar").names
# ['angle', 'area_q1', 'area_q2', 'area_q3', 'area_q4', 'area_ts', 'csi', 'gyration_radius', 'polar_balance']
```

Which means a whole family is one call away:

```python
feats(data, metrics(group="basic").names)
```

To export the catalogue, use `.to_dict()` or `.to_pandas()`, the latter importing pandas only on demand:

```python
metrics().to_pandas()
```

See the [Metrics catalogue](metrics.md) for the full definition of each one.

### Mixing families

Basic and polar metrics can be requested together in a single call. They are computed in one pass:

```python
feats(data, ["mean", "amplitude", "area_ts", "polar_balance"])
```

## Errors

Requesting an unknown metric raises `NotImplementedError`:

```python
feats(data, ["not_a_metric"])
# NotImplementedError: not_a_metric operation is not implemented.
```
