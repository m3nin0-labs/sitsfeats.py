# sitsfeats.py

`sitsfeats.py` extracts features (metrics) from satellite image time series, with operations implemented in `C++` for performance ([Eigen](https://eigen.tuxfamily.org/)) and a small, ergonomic Python API that integrates with the scientific Python stack.

This project is a modern re-implementation of the metrics from the [stmetrics](https://github.com/brazil-data-cube/stmetrics) and [sitsfeats (R)](https://github.com/OldLipe/sitsfeats) projects.

## Installation

To install the `sitsfeats.py` package, please use the following command:

```sh
pip install sitsfeats
```

For more information on the options available for installation, please check the [Installation](installation.md) page.

## Quick example

Everything starts from a 2-D array where each row is one time series. The example below builds a thousand series of 36 observations each:

```python
import numpy as np
from sitsfeats import feats

data = np.random.default_rng(0).normal(5000, 800, size=(1000, 36))

data.shape
# (1000, 36)
```

Now ask for the metrics you want, by name:

```python
result = feats(data, ["mean", "skew", "amplitude"])
```

The result carries the metric names and a stacked matrix, one row per series and one column per metric:

```python
result.names
# ['mean', 'skew', 'amplitude']

result.data.shape
# (1000, 3)
```

Column `j` always corresponds to `result.names[j]`, so the three metrics of the first series read like this:

```python
result.data[0]
# array([ 4.9131e+03, -3.4612e-01,  2.9532e+03])
```

See [Installation](installation.md) to get started, [Usage](usage.md) for the API and the data cube support, and the [Metrics catalogue](metrics.md) for every available metric and its definition.
