#
# Copyright (C) 2024 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Polar metrics."""

from .utils import internal_metric_operation

#
# Metrics names
#
METRICS_NAMES = []

#
# Metrics operations
#
METRICS = {name: internal_metric_operation(name) for name in METRICS_NAMES}
