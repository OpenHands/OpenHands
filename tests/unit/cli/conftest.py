import sys
from pathlib import Path

# Ensure we can import the CLI package from the repo without installing it
REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PKG = REPO_ROOT / 'openhands-cli'
# Add both the project root (so `import openhands_cli` works) and subfolder in case
for p in [REPO_ROOT, CLI_PKG]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
