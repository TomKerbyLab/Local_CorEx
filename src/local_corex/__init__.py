"""Local CorEx public API."""

from .base import BaseCorex, BioCorex, LinearCorex


def partition_data(*args, **kwargs):
	from .partition import partition_data as _partition_data

	return _partition_data(*args, **kwargs)

__all__ = [
	"BaseCorex",
	"BioCorex",
	"LinearCorex",
	"partition_data",
]
