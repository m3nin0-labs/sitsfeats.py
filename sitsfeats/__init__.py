#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""sitsfeats module."""

from .feats import Features, feats
from .registry import metrics, register_builtin_metrics

#
# Version
#
__version__ = "0.2.0"

#
# Register the metrics declared by the compiled extension
#
register_builtin_metrics()

#
# Export the public API
#
__all__ = (
	"__version__",
	"Features",
	"feats",
	"metrics",
)
