"""Conftest for feature test directory.

Ensures step definitions are imported and available for pytest-bdd.
"""

from __future__ import annotations

# Import all step modules and their fixtures directly into namespace
# Wildcard imports ensure pytest-bdd step fixtures are discoverable
from tests.bdd.steps.agent_steps import *  # noqa: F401, F403
from tests.bdd.steps.common_steps import *  # noqa: F401, F403
from tests.bdd.steps.frontend_steps import *  # noqa: F401, F403
from tests.bdd.steps.skills_steps import *  # noqa: F401, F403
