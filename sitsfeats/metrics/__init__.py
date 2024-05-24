#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Metrics module."""

from importlib import import_module

from sitsfeats.registry import MetricsRegistry

#
# Metric modules
#
MODULES = [
    "sitsfeats.metrics.basic",
    "sitsfeats.metrics.polars",
]


#
# Register operation
#
def register_metrics():
    """Register metrics operations."""

    for module in MODULES:
        mod = import_module(module)

        for metric_name in mod.METRICS.keys():
            MetricsRegistry.register(metric_name, mod.METRICS[metric_name])
