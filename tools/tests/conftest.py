"""Make the sibling `tools/` modules importable by their bare names.

The agent-server loads them through `--import-modules` after
`buildAgentServerEnv` puts `tools/` on `OH_EXTRA_PYTHON_PATH`, so tests should
import them the same way rather than via a package path.
"""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
