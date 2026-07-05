#
# Copyright (C) 2024-2026 sitsfeats.py.
#
# sitsfeats.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Metric registry and discovery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class MetricInfo(NamedTuple):
	"""Metadata for a single metric."""

	name: str
	"""Metric name."""

	group: str
	"""Metric family, e.g. ``"basic"`` or ``"polar"``."""

	description: str
	"""One-line summary of what the metric computes."""


class MetricsRegistry:
	"""Registry of available metrics, keyed by name."""

	_metrics: dict[str, MetricInfo] = {}
	"""Dict with metric names as keys and `MetricInfo` records as values."""

	@classmethod
	def register(cls, name: str, group: str, description: str) -> None:
		"""Register a metric.

		Args:
		    name (str): Metric name.

		    group (str): Metric family (e.g. ``"basic"``).

		    description (str): One-line summary.
		"""
		cls._metrics[name] = MetricInfo(name, group, description)

	@classmethod
	def exists(cls, name: str, raise_error: bool = False) -> bool:
		"""Check if a given metric exists.

		Args:
		    name (str): Metric name.

		    raise_error (bool): Raise instead of returning ``False`` when the
		                        metric is unknown.

		Returns:
		    bool: Whether the metric exists.
		"""
		_exists = name in cls._metrics

		if raise_error and not _exists:
			raise NotImplementedError(f"{name} operation is not implemented.")

		return _exists

	@classmethod
	def names(cls) -> list[str]:
		"""List the names of all registered metrics."""
		return list(cls._metrics)

	@classmethod
	def info(cls, name: str) -> MetricInfo:
		"""Return the full `MetricInfo` record for a registered metric.

		Args:
		    name (str): Metric name.

		Returns:
		    MetricInfo: The `MetricInfo` record for the given metric.
		"""
		# check if metric exists
		cls.exists(name, raise_error=True)

		# return metric info
		return cls._metrics[name]

	@classmethod
	def group(cls, name: str) -> str:
		"""Return the family of a registered metric.

		Args:
		    name (str): Metric name.

		Returns:
		    str: The family of the given metric.
		"""
		return cls.info(name).group


class MetricTable(Sequence):
	"""An immutable, dependency-free table of `MetricInfo` records.

	Behaves as a sequence of records (iterate, index, `len`, `in`) and
	renders as an aligned table in the terminal and as an HTML table in Jupyter.

	Use `names` to get the bare metric names (e.g., to pass to `sitsfeats.feats`), or
	`to_dict` / `to_pandas` to export.
	"""

	__slots__ = ("_records",)
	"""Tuple of `MetricInfo` records."""

	def __init__(self, records: Sequence[MetricInfo]) -> None:
		"""Initialize the metric table.

		Args:
		    records (Sequence[MetricInfo]): Sequence of `MetricInfo` records.
		"""
		# convert to tuple
		self._records = tuple(records)

	def __getitem__(self, index: int) -> MetricInfo:  # type: ignore[override]
		"""Get a record by index.

		Args:
			index (int): Index of the record.

		Returns:
			MetricInfo: The `MetricInfo` record at the given index.
		"""
		# The table supports integer indexing only. The Sequence mixin's
		# slice overload is intentionally not implemented.
		return self._records[index]

	def __len__(self) -> int:
		"""Get the number of records in the table.

		Returns:
		    int: The number of records in the table.
		"""
		return len(self._records)

	@property
	def names(self) -> list[str]:
		"""Get the metric names, in table order.

		Returns:
		    list[str]: The metric names, in table order.
		"""
		return [record.name for record in self._records]

	def to_dict(self) -> list[dict[str, str]]:
		"""Return the metric records as a list of dictionaries.

		Returns:
		    list[dict[str, str]]: The metric records as a list of dictionaries.
		"""
		return [record._asdict() for record in self._records]

	def to_pandas(self):
		"""Metric records as a pandas DataFrame.

		Returns:
		    pandas.DataFrame: The metric records as a pandas DataFrame.

		Requires:
		    pandas.DataFrame: The metric records as a pandas DataFrame.
		"""
		import pandas as pd  # noqa: PLC0415 - optional, imported only on demand

		return pd.DataFrame(self._records, columns=list(MetricInfo._fields))

	def __repr__(self) -> str:
		"""Return a string representation of the metric table.

		Returns:
		    str: A string representation of the metric table.
		"""
		# check if the table is empty
		if not self._records:
			return "MetricTable(empty)"

		# get the fields and rows
		fields = MetricInfo._fields

		# get the rows
		rows = [tuple(getattr(r, f) for f in fields) for r in self._records]

		# get the widths
		widths = [
			max(len(fields[i]), *(len(row[i]) for row in rows))
			for i in range(len(fields))
		]

		# clojure - join the columns with the widths
		def line(cols: Sequence[str]) -> str:
			return "  ".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip()

		# join the lines
		return "\n".join([line(fields), *(line(row) for row in rows)])

	def _repr_html_(self) -> str:
		"""Return an HTML representation of the metric table.

		Returns:
		    str: An HTML representation of the metric table.
		"""
		# get the fields
		fields = MetricInfo._fields

		# get the head
		head = "".join(f"<th style='text-align:left'>{f}</th>" for f in fields)

		# get the body
		body = "".join(
			"<tr>"
			+ "".join(
				f"<td style='text-align:left'>{getattr(r, f)}</td>" for f in fields
			)
			+ "</tr>"
			for r in self._records
		)

		# return!
		return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def register_builtin_metrics() -> None:
	"""Register metrics declared by the compiled extension."""
	import _sitsfeats  # noqa: PLC0415 - imported here to keep the C++ name private

	# list metrics
	for name, group, description in _sitsfeats.list_metrics():
		# register metric
		MetricsRegistry.register(name, group, description)


def metrics(group: str | None = None) -> MetricTable:
	"""List the available metrics, with their family and a short description.

	Args:
	    group (str | None): If given, restrict the result to this family
	                        (e.g. ``"basic"`` or ``"polar"``).

	Returns:
	    MetricTable: A table of `MetricInfo` records, sorted by name.

	Example:
	    >>> from sitsfeats import metrics
	    >>> metrics()
	    name       group  description
	    abs_sum    basic  Sum of absolute values
	    ...
	    >>> metrics(group="polar").names
	    ['angle', 'area_q1', ...]
	"""
	# get metric recorsd
	records = [MetricsRegistry.info(name) for name in sorted(MetricsRegistry.names())]

	# filter by group if available
	if group is not None:
		records = [record for record in records if record.group == group]

	# return table
	return MetricTable(records)
