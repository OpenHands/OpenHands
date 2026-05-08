from __future__ import annotations

import subprocess

COMMON_BRANCH_EXAMPLES = "'main', 'feature/foo', or 'release/1.2.3'"


def is_valid_git_branch_name(branch_name: str) -> bool:
    """Return True when branch_name matches git branch naming rules."""
    if not branch_name:
        return False

    return (
        subprocess.run(
            ['git', 'check-ref-format', '--branch', branch_name],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def ensure_valid_git_branch_name(branch_name: str) -> None:
    """Raise ValueError when branch_name is not safe to pass to git checkout."""
    if is_valid_git_branch_name(branch_name):
        return

    raise ValueError(
        f'Invalid git branch name. Common GitHub/GitLab/Bitbucket '
        f'branch names look like {COMMON_BRANCH_EXAMPLES}.'
    )
