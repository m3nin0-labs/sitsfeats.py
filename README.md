## sitsfeats.py ⌛

`sitsfeats.py` is a powerful and efficient Python package designed for extracting a wide range of metrics from satellite image time series. 

> All metrics implemented in this package are derived from the [sitsfeats](https://github.com/OldLipe/sitsfeats) R package.

## Features

The metrics implemented in this version are:

- **Basic**: Basic statistics metrics
- **Polar**: Polar metrics

## Installation

To install the package, you must have [Armadillo](https://arma.sourceforge.net/) installed, then you can use pip:

```bash
pip install git+https://github.com/m3nin0-labs/sitsfeats.py
```

## Usage

The `sitsfeats.py` package is designed for simplicity. It is simple to get started with it. Here's a quick example:

```python
import sitsfeats

ts_median_data = sitsfeats.ts_median(your_numpy_array)
```

## Acknowledgments

We would like to thank the developers and contributors of the `sitsfeats` R package for their work that is the basis of this package.

## Contributing

We welcome contributions! If you have suggestions for improvements or bug fixes, please feel free to fork the repository and submit a pull request.

## License

`sitsfeats.py` is distributed under the MIT license. See LICENSE for more details.
