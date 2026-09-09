"""Startup registration for the standards registry singleton.

Imported via ``--import-modules standards_api`` so plugin discovery is ready
before REST handlers run. Missing or unreadable state dirs are ignored.
"""

from __future__ import annotations

import os
import sys

from standards.api import (  # noqa: F401
    StandardsService,
    handle_request,
    serve_standards,
)
from standards.registry import start_default_registry


def _in_unit_tests() -> bool:
    return "unittest" in sys.modules or "pytest" in sys.modules


if (
    not _in_unit_tests()
    and os.environ.get("OH_STANDARDS_NO_START") != "1"
):
    try:
        start_default_registry()
    except Exception:
        pass
