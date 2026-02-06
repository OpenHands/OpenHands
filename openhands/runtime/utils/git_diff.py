# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
#!/usr/bin/env python3
"""Get git diff in a single git file for the closest git repo in the file system
NOTE: Since this is run as a script, there should be no imports from project files!
"""

import json
import os
import subprocess
import sys
from pathlib import Path

MAX_FILE_SIZE_FOR_GIT_DIFF = 1024 * 1024  # 1 Mb


def get_closest_git_repo(path: Path) -> Path | None:
    while True:
        path = path.parent
        git_path = Path(path, '.git')
        if git_path.is_dir():
            return path
        if path.parent == path:
            return None


def run(cmd: str, cwd: str) -> str:
    result = subprocess.run(
        args=cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    byte_content = result.stderr or result.stdout or b''

    if result.returncode != 0:
        raise RuntimeError(
            f'error_running_cmd:{result.returncode}:{byte_content.decode()}'
        )
    return byte_content.decode().strip()


def get_valid_ref(repo_dir: str) -> str | None:
    refs = []
    try:
        current_branch = run('git --no-pager rev-parse --abbrev-ref HEAD', repo_dir)
        refs.append(f'origin/{current_branch}')
    except RuntimeError:
        pass

    try:
        default_branch = (
            run('git --no-pager remote show origin | grep "HEAD branch"', repo_dir)
            .split()[-1]
            .strip()
        )
        ref_non_default_branch = f'$(git --no-pager merge-base HEAD "$(git --no-pager rev-parse --abbrev-ref origin/{default_branch})")'
        ref_default_branch = f'origin/{default_branch}'
        refs.append(ref_non_default_branch)
        refs.append(ref_default_branch)
    except RuntimeError:
        pass

    # compares with empty tree
    ref_new_repo = (
        '$(git --no-pager rev-parse --verify 4b825dc642cb6eb9a060e54bf8d69288fbee4904)'
    )
    refs.append(ref_new_repo)

    # Find a ref that exists...
    for ref in refs:
        try:
            result = run(f'git --no-pager rev-parse --verify {ref}', repo_dir)
            return result
        except RuntimeError:
            # invalid ref - try next
            continue

    return None


def get_git_diff(relative_file_path: str) -> dict[str, str]:
    path = Path(os.getcwd(), relative_file_path).resolve()
    if os.path.getsize(path) > MAX_FILE_SIZE_FOR_GIT_DIFF:
        raise ValueError('file_to_large')
    closest_git_repo = get_closest_git_repo(path)
    if not closest_git_repo:
        raise ValueError('no_repository')
    current_rev = get_valid_ref(str(closest_git_repo))
    try:
        original = run(
            f'git show "{current_rev}:{path.relative_to(closest_git_repo)}"',
            str(closest_git_repo),
        )
    except RuntimeError:
        original = ''
    try:
        with open(path, 'r') as f:
            modified = '\n'.join(f.read().splitlines())
    except FileNotFoundError:
        modified = ''
    return {
        'modified': modified,
        'original': original,
    }


if __name__ == '__main__':
    diff = get_git_diff(sys.argv[-1])
    print(json.dumps(diff))
