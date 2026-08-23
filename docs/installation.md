# Installation

Install `sitsfeats.py` from PyPI:

```bash
pip install sitsfeats
```

To also work with labelled data cubes, install the `xarray` extra, which brings in `xarray` and `dask`:

```bash
pip install "sitsfeats[xarray]"
```

The cube support is described in [Working with data cubes](usage.md#working-with-data-cubes).

## What gets installed

The package requires Python 3.11 or newer and depends only on `numpy`.

Its operations are implemented in `C++`, so a binary wheel is used when one is available for your platform. When there is none, `pip` builds the extension from source. That build needs a `C++ 17` compiler, but no system libraries: the extension uses [nanobind](https://github.com/wjakob/nanobind) and the header-only [Eigen](https://eigen.tuxfamily.org/) library, both resolved automatically at build time.

## uv (development)

The project is developed with [uv](https://docs.astral.sh/uv/). Clone the repository and sync, which builds the C++ extension and installs the dev tools:

```bash
git clone https://github.com/m3nin0-labs/sitsfeats.py
cd sitsfeats.py
uv sync
```

Common tasks:

```bash
uv run pytest          # run the test suite
uvx ruff check .       # lint
uv run mkdocs serve    # preview these docs
uv build               # build wheel + sdist
```
