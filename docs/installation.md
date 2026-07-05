# Installation

`sitsfeats.py` builds from source with **no system dependencies**. The C++ extension uses [nanobind](https://github.com/wjakob/nanobind) and the header-only [Eigen](https://eigen.tuxfamily.org/) library, both resolved automatically at build time.

## pip

```bash
pip install sitsfeats.py # or git+https://github.com/m3nin0-labs/sitsfeats.py
```

To enable the [xarray integration](xarray.md):

```bash
pip install "sitsfeats.py[xarray]"
```

## uv (development)

The project is developed with [uv](https://docs.astral.sh/uv/). Clone the repository and sync, sync this builds the C++ extension and installs the dev tools:

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

## Requirements

- Python ≥ 3.11
- A C++17 compiler (only needed to build from source; binary wheels require none)
