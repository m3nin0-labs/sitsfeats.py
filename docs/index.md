# sitsfeats.py

`sitsfeats.py` extracts features (matrics) from satellite image time series, with operations implemented in `C++` for performance ([Eigen](https://eigen.tuxfamily.org/)) and a small, ergonomic Python API that integrates with the scientific Python stack.

This project is a modern re-implementation of the metrics from the [stmetrics](https://github.com/brazil-data-cube/stmetrics) and [sitsfeats (R)](https://github.com/OldLipe/sitsfeats) projects.

## Installation

To install the `sitsfeats.py` package, please use the following command:

```sh
pip install sitsfeats.py
```

For more information on the options available for installation, please check the [Installation](installation.md) page.

## Quick example

Below there is one quick example showing how easy is to use the `sitsfeats.py` API.

```python
import numpy as np
from sitsfeats import feats

# define data
data = np.random.default_rng(0).normal
  (5000, 800, size=(1000, 36)
)

# features!
result = feats(data, ["mean", "skew", "amplitude"])

# check the result
result.names      # ['mean', 'skew', 'amplitude']
result.data.shape # (1000, 3)
```

See [Installation](installation.md) to get started, [Usage](usage.md) for the API, and the [Metrics catalogue](metrics.md) for every available metric and its definition.
