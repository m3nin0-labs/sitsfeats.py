# sitsfeats.py ⌛

Python package for extracting metrics from satellite image time series. 

### Installation

The package builds from source with no system dependencies. The C++ extension uses [nanobind](https://github.com/wjakob/nanobind) and the header-only [Eigen](https://eigen.tuxfamily.org/) library, both resolved automatically at build time:

```bash
pip install sitsfeats
```

### Usage

The `sitsfeats.py` package is designed for simplicity. It is simple to get started with it. Here's a quick example:

```python
from sitsfeats import feats

# Each row of the input array is a time-series
result = feats(your_numpy_data, ['median', 'skew'])

result.names
#> ['median', 'skew']

# A stacked (n_series, n_metrics) array; column j is result.names[j]
result.data
#> array([[ 5.5850e+03, -8.2705e-01],
#>        [ 5.0490e+03,  3.9131e-02],
#>        [ 6.6015e+03, -3.6274e-01],
#>        [ 6.0470e+03, -3.0612e-01],
#>        [ 4.6960e+03,  3.4300e-01]])

# Or a {name: column} mapping.
result.to_dict()
```

#### Working with data cubes (xarray)

The `feats` function also supports `xarray` data as input. To use this feature, first install the `xarray` dependencies:

```shell
pip install sitsfeats[xarray]
```

Then, you can use the `xarray` data as input:

```python
from sitsfeats import feats

# cube: an xarray.DataArray with dims like (time, y, x)
features = feats(cube, ["mean", "skew", "iqr"], dim="time")

# A Dataset with one variable per metric, each shaped (y, x)
features["skew"]
```

To learn more, work through the step-by-step [jupytext](https://jupytext.readthedocs.io) tutorials in the `examples` directory (`ts-numpy.py` for arrays and `ts-xarray.py` for data cubes - run them as scripts or open them as notebooks). Check also the [documentation](https://m3nin0-labs.github.io/sitsfeats.py/).

### Metrics available

To see the metrics available in `sitsfeats.py`, it is possible to use the command `metrics`:

```python
from sitsfeats import metrics

metrics()               # all metrics
metrics(group="polar")  # just one family

# name             group  description
# abs_sum          basic  Sum of absolute values
# amd              basic  Mean absolute first difference
# amplitude        basic  Range (max - min)
# ...
# area_ts          polar  Area of the polar-plot polygon
# polar_balance    polar  Std-dev of the four quadrant areas
```

### Development

The project uses [uv](https://docs.astral.sh/uv/). After cloning:

```bash
uv sync            # builds the C++ extension and installs dev tools
uv run pytest      # run the test suite
uvx ruff check .   # lint
uv run mkdocs serve  # preview the docs
```

### Contributing

We welcome contributions! If you have suggestions for improvements or bug fixes, please feel free to fork the repository and submit a pull request.

### Acknowledgments

We would like to thank the developers and contributors of the [sitsfeats (R)](https://github.com/OldLipe/sitsfeats) and [stmetrics](https://github.com/brazil-data-cube/stmetrics) for their work that is the basis of this package.

### License

`sitsfeats.py` is distributed under the MIT license. See [LICENSE](https://github.com/m3nin0-labs/sitsfeats.py/blob/main/LICENSE) for more details.
