"""Startup registration for the codebase-graph singleton.

Imported via ``--import-modules graph_indexer`` so the SQLite store is ready
before graph API handlers run. Missing or unreadable state dirs are ignored.
"""

from __future__ import annotations

import os
import sys

from codebase_graph import GraphStore, default_db_path, get_active_store, set_active_store


def start_default_store() -> GraphStore | None:
    existing = get_active_store()
    if existing is not None:
        return existing
    try:
        store = GraphStore(default_db_path())
        set_active_store(store)
        return store
    except Exception:
        return None


def _in_unit_tests() -> bool:
    return "unittest" in sys.modules or "pytest" in sys.modules


if (
    not _in_unit_tests()
    and os.environ.get("OH_GRAPH_INDEXER_NO_START") != "1"
):
    try:
        start_default_store()
    except Exception:
        pass
