#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""feats module."""


import numpy as np

from sitsfeats.registry import MetricsRegistry


#
# Utility
#
def _check_metrics(metrics: list[str]) -> None:
    """Check if given metrics are available.

    Args:
        metrics (List[str]): Metric names.

    Raises:
        ValueError: If metric is not available, an
                    `ValueError` is raised.
    """
    for metric in metrics:
        MetricsRegistry.exists(metric, raise_error=True)


#
# Feats operation
#
def feats(metrics: list[str], data: np.ndarray) -> dict:
    """Generate metrics (features) from satellite image time-series.

    Arts:
        metrics (List[str]): Metric names.

        data (np.ndarray): Time-series data. Each row is interpreted
                           as a time-series.

    Returns:
        dict: Metrics results as a dict.
    """
    _check_metrics(metrics)

    result = {}
    for metric in metrics:
        result[metric] = MetricsRegistry.apply(metric, data)

    return result
