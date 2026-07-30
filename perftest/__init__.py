"""Capturing Hermes-3 performance test results into a durable record.

Deliberately separate from any particular results store: this package knows how
to turn a finished case directory into a bundle plus one index row, and nothing
about whose results they are. The store itself -- the index file, the bundles,
the conventions for naming and epochs -- lives in its own repository.
"""

from .extract import extract_case
from .index import INDEX_COLUMNS, read_index, write_index

__all__ = ["extract_case", "read_index", "write_index", "INDEX_COLUMNS"]
