# Data cubes (xarray)

With the optional `xarray` extra you can extract metrics directly from a labelled data cube, preserving its spatial coordinates and attributes.

```bash
pip install "sitsfeats.py[xarray]"
```

## `feats` on a cube

The same `feats` entry point handles cubes, just pass an xarray object as the first argument. It dispatches on the input type and returns an `xarray.Dataset`:

```python
import xarray as xr
from sitsfeats import feats

# cube: an xarray.DataArray with a time dimension, e.g. dims (time, y, x)
features = feats(cube, ["mean", "skew", "iqr"], dim = "time")
```

The result is an `xarray.Dataset` with one variable per metric, each shaped like the cube's spatial dimensions:

```python
features                      # Dataset with data vars: mean, skew, iqr
features["skew"]              # DataArray with dims (y, x)
features["y"], features["x"]  # original coordinates, preserved
```

## Dask: parallel & out-of-core

The cube path is built on `xarray.apply_ufunc` with `dask="parallelized"`. If the cube is chunked, computation stays lazy and runs per-chunk. In parallel and without loading the whole cube into memory:

```python
chunked = cube.chunk({"y": 256, "x": 256})   # keep `time` in a single chunk
features = feats(chunked, ["mean", "amplitude"])

features["mean"].data    # a dask array — nothing computed yet
result = features.compute()
```

!!! note
    The time dimension must not be chunked (it is the reduction axis). Chunk the
    spatial dimensions only.

## Datasets

A `Dataset` input is processed per variable; output variables are named `<variable>_<metric>`:

```python
ds = xr.Dataset({"ndvi": ndvi_cube, "evi": evi_cube})
feats(ds, ["mean"])     # data vars: ndvi_mean, evi_mean
```
