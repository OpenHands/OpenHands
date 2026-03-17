#!/usr/bin/env python3
"""
Find all PRs that went in between two commits in the OpenHands/OpenHands repository.
Handles cherry-picks and different merge strategies.

This script is designed to run from within the OpenHands repository under .github/scripts:
    .github/scripts/find_prs_between_commits.py

Usage: find_prs_between_commits <older_commit> <newer_commit> [--repo <path>]
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


def find_openhands_repo() -> Optional[Path]:
    """
    Find the OpenHands repository.
    Since this script is designed to live in .github/scripts/, it assumes
    the repository root is two levels up from the script location.
    Tries:
    1. Repository root (../../ from script location)
    2. Current directory
    3. Environment variable OPENHANDS_REPO
    """
    # Check repository root (assuming script is in .github/scripts/)
    script_dir = Path(__file__).parent.absolute()
    repo_root = (
        script_dir.parent.parent
    )  # Go up two levels: scripts -> .github -> repo root
    if (repo_root / '.git').exists():
        return repo_root

    # Check current directory
    if (Path.cwd() / '.git').exists():
        return Path.cwd()

    # Check environment variable
    if 'OPENHANDS_REPO' in os.environ:
        repo_path = Path(os.environ['OPENHANDS_REPO'])
        if (repo_path / '.git').exists():
            return repo_path

    return None


def run_git_command(cmd: list[str], repo_path: Path) -> str:
    """Run a git command in the repository directory and return its output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=str(repo_path)
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f'Error running git command: {" ".join(cmd)}', file=sys.stderr)
        print(f'Error: {e.stderr}', file=sys.stderr)
        sys.exit(1)


def extract_pr_numbers_from_message(message: str) -> set[int]:
    """Extract PR numbers from commit message in any common format."""
    # Match #12345 anywhere, including in patterns like (#12345) or "Merge pull request #12345"
    matches = re.findall(r'#(\d+)', message)
    return set(int(m) for m in matches)


def get_commit_info(commit_hash: str, repo_path: Path) -> tuple[str, str, str]:
    """Get commit subject, body, and author from a commit hash."""
    subject = run_git_command(
        ['git', 'log', '-1', '--format=%s', commit_hash], repo_path
    )
    body = run_git_command(['git', 'log', '-1', '--format=%b', commit_hash], repo_path)
    author = run_git_command(
        ['git', 'log', '-1', '--format=%an <%ae>', commit_hash], repo_path
    )
    return subject, body, author


def get_commits_between(
    older_commit: str, newer_commit: str, repo_path: Path
) -> list[str]:
    """Get all commit hashes between two commits."""
    commits_output = run_git_command(
        ['git', 'rev-list', f'{older_commit}..{newer_commit}'], repo_path
    )

    if not commits_output:
        return []

    return commits_output.split('\n')


def get_pr_info_from_github(pr_number: int, repo_path: Path) -> Optional[dict]:
    """Get PR information from GitHub API if GITHUB_TOKEN is available."""
    try:
        # Set up environment with GitHub token
        env = os.environ.copy()
        if 'GITHUB_TOKEN' in env:
            env['GH_TOKEN'] = env['GITHUB_TOKEN']

        result = subprocess.run(
            [
                'gh',
                'pr',
                'view',
                str(pr_number),
                '--json',
                'number,title,author,mergedAt,baseRefName,headRefName,url',
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(repo_path),
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


def find_prs_between_commits(
    older_commit: str, newer_commit: str, repo_path: Path
) -> dict[int, dict]:
    """
    Find all PRs that went in between two commits.
    Returns a dictionary mapping PR numbers to their information.
    """
    print(f'Repository: {repo_path}', file=sys.stderr)
    print('Finding PRs between commits:', file=sys.stderr)
    print(f'  Older: {older_commit}', file=sys.stderr)
    print(f'  Newer: {newer_commit}', file=sys.stderr)
    print(file=sys.stderr)

    # Verify commits exist
    try:
        run_git_command(['git', 'rev-parse', '--verify', older_commit], repo_path)
        run_git_command(['git', 'rev-parse', '--verify', newer_commit], repo_path)
    except SystemExit:
        print('Error: One or both commits not found in repository', file=sys.stderr)
        sys.exit(1)

    # Extract PRs from the older commit itself (to exclude from results)
    # These PRs are already included at or before the older commit
    older_subject, older_body, _ = get_commit_info(older_commit, repo_path)
    older_message = f'{older_subject}\n{older_body}'
    excluded_prs = extract_pr_numbers_from_message(older_message)

    if excluded_prs:
        print(
            f'Excluding PRs already in older commit: {", ".join(f"#{pr}" for pr in sorted(excluded_prs))}',
            file=sys.stderr,
        )
        print(file=sys.stderr)

    # Get all commits between the two
    commits = get_commits_between(older_commit, newer_commit, repo_path)
    print(f'Found {len(commits)} commits to analyze', file=sys.stderr)
    print(file=sys.stderr)

    # Extract PR numbers from all commits
    pr_info: dict[int, dict] = {}
    commits_by_pr: dict[int, list[str]] = defaultdict(list)

    for commit_hash in commits:
        subject, body, author = get_commit_info(commit_hash, repo_path)
        full_message = f'{subject}\n{body}'

        pr_numbers = extract_pr_numbers_from_message(full_message)

        for pr_num in pr_numbers:
            # Skip PRs that are already in the older commit
            if pr_num in excluded_prs:
                continue

            commits_by_pr[pr_num].append(commit_hash)

            if pr_num not in pr_info:
                pr_info[pr_num] = {
                    'number': pr_num,
                    'first_commit': commit_hash[:8],
                    'first_commit_subject': subject,
                    'commits': [],
                    'github_info': None,
                }

            pr_info[pr_num]['commits'].append(
                {'hash': commit_hash[:8], 'subject': subject, 'author': author}
            )

    # Try to get additional info from GitHub API
    print('Fetching additional info from GitHub API...', file=sys.stderr)
    for pr_num in pr_info.keys():
        github_info = get_pr_info_from_github(pr_num, repo_path)
        if github_info:
            pr_info[pr_num]['github_info'] = github_info

    print(file=sys.stderr)

    return pr_info


def print_results(pr_info: dict[int, dict]):
    """Print the results in a readable format."""
    sorted_prs = sorted(pr_info.items(), key=lambda x: x[0])

    print(f'{"=" * 80}')
    print(f'Found {len(sorted_prs)} PRs')
    print(f'{"=" * 80}')
    print()

    for pr_num, info in sorted_prs:
        print(f'PR #{pr_num}')

        if info['github_info']:
            gh = info['github_info']
            print(f'  Title: {gh["title"]}')
            print(f'  Author: {gh["author"]["login"]}')
            print(f'  URL: {gh["url"]}')
            if gh.get('mergedAt'):
                print(f'  Merged: {gh["mergedAt"]}')
            if gh.get('baseRefName'):
                print(f'  Base: {gh["baseRefName"]} ← {gh["headRefName"]}')
        else:
            print(f'  Subject: {info["first_commit_subject"]}')

        # Show if this PR has multiple commits (cherry-picked or multiple commits)
        commit_count = len(info['commits'])
        if commit_count > 1:
            print(
                f'  ⚠️  Found {commit_count} commits (possible cherry-pick or multi-commit PR):'
            )
            for commit in info['commits'][:3]:  # Show first 3
                print(f'      {commit["hash"]}: {commit["subject"][:60]}')
            if commit_count > 3:
                print(f'      ... and {commit_count - 3} more')
        else:
            print(f'  Commit: {info["first_commit"]}')

        print()


def main():
    if len(sys.argv) < 3:
        print('Usage: find_prs_between_commits <older_commit> <newer_commit> [options]')
        print()
        print('Arguments:')
        print('  <older_commit>  The older commit hash (or ref)')
        print('  <newer_commit>  The newer commit hash (or ref)')
        print()
        print('Options:')
        print('  --json          Output results in JSON format')
        print('  --repo <path>   Path to OpenHands repository (default: auto-detect)')
        print()
        print('Example:')
        print(
            '  find_prs_between_commits c79e0cd3c7a2501a719c9296828d7a31e4030585 35bddb14f15124a3dc448a74651a6592911d99e9'
        )
        print()
        print('Repository Detection:')
        print('  The script will try to find the OpenHands repository in this order:')
        print('  1. --repo argument')
        print('  2. Repository root (../../ from script location)')
        print('  3. Current directory')
        print('  4. OPENHANDS_REPO environment variable')
        print()
        print('Environment variables:')
        print(
            '  GITHUB_TOKEN    Optional. If set, will fetch additional PR info from GitHub API'
        )
        print('  OPENHANDS_REPO  Optional. Path to OpenHands repository')
        sys.exit(1)

    older_commit = sys.argv[1]
    newer_commit = sys.argv[2]
    json_output = '--json' in sys.argv

    # Check for --repo argument
    repo_path = None
    if '--repo' in sys.argv:
        repo_idx = sys.argv.index('--repo')
        if repo_idx + 1 < len(sys.argv):
            repo_path = Path(sys.argv[repo_idx + 1])
            if not (repo_path / '.git').exists():
                print(f'Error: {repo_path} is not a git repository', file=sys.stderr)
                sys.exit(1)

    # Auto-detect repository if not specified
    if repo_path is None:
        repo_path = find_openhands_repo()
        if repo_path is None:
            print('Error: Could not find OpenHands repository', file=sys.stderr)
            print('Please either:', file=sys.stderr)
            print(
                '  1. Place this script in .github/scripts/ within the OpenHands repository',
                file=sys.stderr,
            )
            print('  2. Run from the OpenHands repository directory', file=sys.stderr)
            print(
                '  3. Use --repo <path> to specify the repository location',
                file=sys.stderr,
            )
            print('  4. Set OPENHANDS_REPO environment variable', file=sys.stderr)
            sys.exit(1)

    # Find PRs
    pr_info = find_prs_between_commits(older_commit, newer_commit, repo_path)

    if json_output:
        # Output as JSON
        print(json.dumps(pr_info, indent=2))
    else:
        # Print results in human-readable format
        print_results(pr_info)

        # Also print a simple list for easy copying
        print(f'{"=" * 80}')
        print('PR Numbers (for easy copying):')
        print(f'{"=" * 80}')
        sorted_pr_nums = sorted(pr_info.keys())
        print(', '.join(f'#{pr}' for pr in sorted_pr_nums))


if __name__ == '__main__':
    main()
