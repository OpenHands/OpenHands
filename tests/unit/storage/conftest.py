"""Windows Python 3.12 compatibility shim for openhands.storage tests.

On Windows with Python 3.12, ``google.api_core.__init__`` calls
``importlib.metadata.packages_distributions()`` during import.  That call
iterates over every installed distribution's file list via the ``importlib_metadata``
backport, which crashes with a ``KeyboardInterrupt`` / ``pathlib.Path`` error
when a distribution entry contains a path fragment that triggers a Windows-specific
bug in the bundled ``importlib_metadata``.

pytest intercepts the ``KeyboardInterrupt`` and reports the whole
``openhands.storage`` package as unimportable (``ModuleNotFoundError``).

The fix: patch ``importlib.metadata.packages_distributions`` to return an empty
mapping *before* the first ``import openhands.storage`` executes.  The version
check in ``google.api_core`` gracefully degrades when the package name cannot
be resolved — it simply skips the deprecation warning — so the rest of the
package initialises normally.

This conftest is loaded at session start for all ``tests/unit/storage/`` tests,
making the shim available in time for collection.
"""
from __future__ import annotations

import importlib.metadata
import sys
from unittest.mock import patch

# Only apply the shim when the stdlib metadata call is known to crash.
# We detect this by checking whether importlib_metadata (the backport) is
# present, since it is the proximate cause of the Windows path crash.
_backport_present = 'importlib_metadata' in sys.modules or any(
    'importlib_metadata' in str(p) for p in sys.path
)
print(f"[conftest] sys.platform={sys.platform!r} _backport_present={_backport_present}")

if sys.platform == 'win32' and _backport_present:
    _patcher = patch.object(
        importlib.metadata,
        'packages_distributions',
        return_value={},
    )
    _patcher.start()
