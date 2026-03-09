# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
#!/usr/bin/env python3
"""Get git changes in the current working directory relative to the remote origin if possible.
NOTE: Since this is run as a script, there should be no imports from project files!
"""

import glob
import json
import os
import subprocess
from pathlib import Path


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


def get_changes_in_repo(repo_dir: str) -> list[dict[str, str]]:
    # Gets the status relative to the origin default branch - not the same as `git status`

    ref = get_valid_ref(repo_dir)
    if not ref:
        return []

    # Get changed files
    changed_files = run(
        f'git --no-pager diff --name-status {ref}', repo_dir
    ).splitlines()
    changes = []
    for line in changed_files:
        if not line.strip():
            raise RuntimeError(f'unexpected_value_in_git_diff:{changed_files}')

        # Handle different output formats from git diff --name-status
        # Depending on git config, format can be either:
        # * "A file.txt"
        # * "A       file.txt"
        # * "R100    old_file.txt    new_file.txt" (rename with similarity percentage)
        parts = line.split()
        if len(parts) < 2:
            raise RuntimeError(f'unexpected_value_in_git_diff:{changed_files}')

        status = parts[0].strip()

        # Handle rename operations (status starts with 'R' followed by similarity percentage)
        if status.startswith('R') and len(parts) == 3:
            # Rename: convert to delete (old path) + add (new path)
            old_path = parts[1].strip()
            new_path = parts[2].strip()
            changes.append(
                {
                    'status': 'D',
                    'path': old_path,
                }
            )
            changes.append(
                {
                    'status': 'A',
                    'path': new_path,
                }
            )
            continue

        # Handle copy operations (status starts with 'C' followed by similarity percentage)
        elif status.startswith('C') and len(parts) == 3:
            # Copy: only add the new path (original remains)
            new_path = parts[2].strip()
            changes.append(
                {
                    'status': 'A',
                    'path': new_path,
                }
            )
            continue

        # Handle regular operations (M, A, D, etc.)
        elif len(parts) == 2:
            path = parts[1].strip()
        else:
            raise RuntimeError(f'unexpected_value_in_git_diff:{changed_files}')

        if status == '??':
            status = 'A'
        elif status == '*':
            status = 'M'

        # Check for valid single-character status codes
        if status in {'M', 'A', 'D', 'U'}:
            changes.append(
                {
                    'status': status,
                    'path': path,
                }
            )
        else:
            raise RuntimeError(f'unexpected_status_in_git_diff:{changed_files}')

    # Get untracked files
    untracked_files = run(
        'git --no-pager ls-files --others --exclude-standard', repo_dir
    ).splitlines()
    for path in untracked_files:
        if path:
            changes.append({'status': 'A', 'path': path})

    return changes


def get_git_changes(cwd: str) -> list[dict[str, str]]:
    git_dirs = {
        os.path.dirname(f)[2:]
        for f in glob.glob('./*/.git', root_dir=cwd, recursive=True)
    }

    # First try the workspace directory
    changes = get_changes_in_repo(cwd)

    # Filter out any changes which are in one of the git directories
    changes = [
        change
        for change in changes
        if next(
            iter(git_dir for git_dir in git_dirs if change['path'].startswith(git_dir)),
            None,
        )
        is None
    ]

    # Add changes from git directories
    for git_dir in git_dirs:
        git_dir_changes = get_changes_in_repo(str(Path(cwd, git_dir)))
        for change in git_dir_changes:
            change['path'] = git_dir + '/' + change['path']
            changes.append(change)

    changes.sort(key=lambda change: change['path'])

    return changes


if __name__ == '__main__':
    try:
        changes = get_git_changes(os.getcwd())
        print(json.dumps(changes))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
