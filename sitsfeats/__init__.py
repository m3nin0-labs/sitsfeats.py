#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""sitsfeats module."""

from .feats import feats
from .metrics import register_metrics

#
# Version
#
__version__ = "0.1.0"

#
# Register metrics operations.
#
register_metrics()


__all__ = (
    "__version__",
    "feats",
)
