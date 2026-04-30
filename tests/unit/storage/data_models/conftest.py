"""Windows Python 3.12 compatibility shim for openhands.storage.data_models tests.

On Windows with Python 3.12, importing openhands.storage triggers
google.api_core.__init__ which calls packages_distributions(); that function
raises KeyboardInterrupt via pathlib on certain Windows environments.
KeyboardInterrupt is a BaseException so it escapes the ``except Exception``
guard in _get_pypi_package_name, crashes the import of openhands.storage,
and pytest then reports ModuleNotFoundError.

Fix applied at conftest.py collection time (before any test module is imported):

1. Add the project root to sys.path so openhands namespace packages
   (openhands.core, openhands.integrations, openhands.events, etc.) are
   resolvable via the normal import machinery.

2. Stub openhands.storage in sys.modules to bypass __init__.py (which
   contains the ``from openhands.storage.google_cloud import …`` that
   triggers the google.api_core crash).  __path__ is set so sub-package
   imports (openhands.storage.data_models.*, openhands.storage.files, etc.)
   still resolve via the real filesystem.

3. Expose FileStore and get_file_store on the stub because several modules
   in the transitive import chain do
   ``from openhands.storage import FileStore / get_file_store``
   at module level (openhands.core.config.utils, openhands.events.stream).

4. Pre-stub openhands.storage.google_cloud so it can never be accidentally
   imported through the stub's __path__.
"""

from __future__ import annotations

import pathlib
import sys
import types

if sys.platform == 'win32':
    _root = pathlib.Path(__file__).parents[4]  # project root: c:\openhands

    # 1) Ensure the project root is on sys.path so that openhands.core,
    #    openhands.integrations, openhands.events, openhands.utils, etc.
    #    are findable as namespace packages without relying on PYTHONPATH.
    _root_str = str(_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

    if 'openhands.storage' not in sys.modules:
        # 2) Try the real import first.  On most Windows machines this works
        #    fine.  The stub is only needed on specific environments where
        #    google.api_core.__init__ calls packages_distributions() via pathlib
        #    and that function raises KeyboardInterrupt, crashing the import.
        #    Applying the stub unconditionally poisons sys.modules for every
        #    other test collected in the same pytest session (e.g. app_server
        #    tests that rely on the real get_file_store).
        try:
            import openhands.storage  # noqa: F401  # triggers real __init__.py

        except (KeyboardInterrupt, SystemExit, ImportError):
            # Real import failed – fall back to the stub so that data_models
            # tests can still be collected and run.
            from unittest.mock import MagicMock

            _stub = types.ModuleType('openhands.storage')
            _stub.__path__ = [  # type: ignore[attr-defined]
                str(_root / 'openhands' / 'storage')
            ]
            _stub.__package__ = 'openhands.storage'

            # Re-export the two symbols imported at module level elsewhere.
            _stub.get_file_store = MagicMock()  # type: ignore[attr-defined]
            _stub.FileStore = MagicMock  # type: ignore[attr-defined]

            sys.modules['openhands.storage'] = _stub

            # Pre-stub google_cloud to prevent accidental import via __path__.
            _gc_stub = types.ModuleType('openhands.storage.google_cloud')
            _gc_stub.GoogleCloudFileStore = MagicMock  # type: ignore[attr-defined]
            sys.modules['openhands.storage.google_cloud'] = _gc_stub
