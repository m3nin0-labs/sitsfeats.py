#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Metric utilities."""

import _sitsfeats


def internal_metric_name(base_name: str) -> str:
    """Internal metric names."""
    return f"C_ts_{base_name}"


def internal_metric_operation(base_name: str) -> callable:
    """Get internal metric.

    Args:
        base_name (str): Base metric name.

    Returns:
        callable: Metric function.
    """
    return getattr(_sitsfeats, internal_metric_name(base_name))
