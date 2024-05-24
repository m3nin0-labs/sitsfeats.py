#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Metric registry module."""

import numpy as np


class MetricsRegistry:
    """Metrics registry class."""

    _metrics = {}
    """Metrics storage."""

    @classmethod
    def register(cls, name: str, fnc: callable) -> None:
        """Register a metric with its callable function.

        Args:
            name (str): Metric name

            fnc (callable): Metric callable function.
        """
        cls._metrics[name] = fnc

    @classmethod
    def exists(cls, name: str, raise_error: bool = False) -> bool:
        """Check if a given metric exists.

        Args:
            name (str): Metric name.

            raise_error (bool): Flag indicating if an error should be raised
                                if the given metric doesn't exist.

        Returns:
            bool: Flag indicating if the given metric exists.
        """
        _exists = name in cls._metrics

        if raise_error and not _exists:
            raise NotImplementedError(f"{name} operation is not implemented.")

        return _exists

    @classmethod
    def apply(cls, name: str, data: np.ndarray):
        """Apply a given metric to a data.

        Args:
            name (str): Metric name.

            data (np.ndarray): Data to apply metric.

        Returns:
            np.ndarray: Metric result.
        """
        cls.exists(name, True)

        return cls._metrics[name](data)
