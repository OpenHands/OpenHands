"""
Regression test for run_budget_maintenance.py's import style.

The production Docker image (enterprise/Dockerfile) does
`COPY --chown=... enterprise .` into WORKDIR /app, which flattens the
contents of the `enterprise/` directory directly into /app instead of
preserving an `enterprise/` subdirectory. This means there is no
importable top-level `enterprise` package in the deployed container --
sibling modules like `run_maintenance_tasks` must be imported unqualified,
exactly like the existing `maintenance-tasks-cronjob.yaml` invokes
`python -m run_maintenance_tasks` (not `python -m enterprise.run_maintenance_tasks`).

If `run_budget_maintenance.py` ever imports something as `enterprise.foo`
or `from enterprise import foo`, the budget-maintenance cronjob will crash
with `ModuleNotFoundError: No module named 'enterprise'` in production,
even though it works fine in local/monorepo dev environments where a
`.pth` file on sys.path makes the repo root's `enterprise` directory
importable as a package.
"""

import ast
from pathlib import Path

RUN_BUDGET_MAINTENANCE_PATH = (
    Path(__file__).resolve().parents[2] / "run_budget_maintenance.py"
)


def _top_level_import_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_run_budget_maintenance_does_not_import_enterprise_package():
    """`enterprise` is not an importable package inside the production
    container (its contents are flattened directly into /app), so
    run_budget_maintenance.py must never do `import enterprise` or
    `from enterprise import ...`."""
    source = RUN_BUDGET_MAINTENANCE_PATH.read_text()
    imported_names = _top_level_import_names(source)
    assert "enterprise" not in imported_names, (
        "run_budget_maintenance.py must not import the 'enterprise' "
        "top-level package -- it is not importable in the deployed "
        "container. Use an unqualified import (e.g. "
        "`import run_maintenance_tasks`) instead."
    )
