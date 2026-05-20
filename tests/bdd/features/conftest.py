"""Conftest for feature test directory.

Ensures step definitions are imported and available for pytest-bdd.
"""

from __future__ import annotations

# Import step modules for pytest-bdd discovery
from tests.bdd.steps import agent_steps  # noqa: F401
from tests.bdd.steps import common_steps  # noqa: F401
from tests.bdd.steps import frontend_steps  # noqa: F401
