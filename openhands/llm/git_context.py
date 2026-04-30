"""Git context provider for injecting repository information into LLM prompts."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitContextProvider:
    """Provides Git repository context for LLM prompts."""

    def __init__(self, repo_path: Optional[str] = None):
        """Initialize the Git context provider.

        Args:
            repo_path: Path to the Git repository. If None, uses current directory.
        """
        self.repo_path = Path(repo_path or '.')

    def is_git_repo(self) -> bool:
        """Check if the current path is a Git repository.

        Returns:
            True if .git directory exists, False otherwise
        """
        git_dir = self.repo_path / '.git'
        return git_dir.exists()

    def get_current_branch(self) -> Optional[str]:
        """Get the current Git branch name.

        Returns:
            Branch name or None if not in a Git repository
        """
        if not self.is_git_repo():
            return None

        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f'Failed to get Git branch: {e}')

        return None

    def get_staged_files(self) -> Optional[list[str]]:
        """Get list of staged files in Git.

        Returns:
            List of staged file paths or None if not in a Git repository
        """
        if not self.is_git_repo():
            return None

        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f'Failed to get staged files: {e}')

        return None

    def get_uncommitted_changes(self) -> Optional[dict[str, int]]:
        """Get count of uncommitted changes (modified, added, deleted).

        Returns:
            Dictionary with 'modified', 'added', 'deleted' counts, or None if not in a Git repository
        """
        if not self.is_git_repo():
            return None

        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                modified = sum(
                    1 for line in result.stdout.split('\n') if line.startswith(' M')
                )
                added = sum(
                    1 for line in result.stdout.split('\n') if line.startswith('??')
                )
                deleted = sum(
                    1 for line in result.stdout.split('\n') if line.startswith(' D')
                )
                return {'modified': modified, 'added': added, 'deleted': deleted}
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f'Failed to get uncommitted changes: {e}')

        return None

    def get_recent_commits(self, count: int = 3) -> Optional[list[str]]:
        """Get recent commit messages.

        Args:
            count: Number of recent commits to retrieve (default 3)

        Returns:
            List of recent commit messages or None if not in a Git repository
        """
        if not self.is_git_repo():
            return None

        try:
            result = subprocess.run(
                ['git', 'log', f'-{count}', '--oneline'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f'Failed to get recent commits: {e}')

        return None

    def get_repo_context(self) -> Optional[str]:
        """Get a summary of Git repository context.

        Returns:
            Formatted Git context string or None if not in a Git repository
        """
        if not self.is_git_repo():
            return None

        try:
            context_parts = []

            # Current branch
            branch = self.get_current_branch()
            if branch:
                context_parts.append(f'Current branch: {branch}')

            # Staged files
            staged = self.get_staged_files()
            if staged:
                staged_str = ', '.join(staged[:5])  # Limit to first 5
                if len(staged) > 5:
                    staged_str += f' ... and {len(staged) - 5} more'
                context_parts.append(f'Staged files: {staged_str}')

            # Uncommitted changes
            changes = self.get_uncommitted_changes()
            if changes and sum(changes.values()) > 0:
                changes_list = [f'{k}: {v}' for k, v in changes.items() if v > 0]
                context_parts.append(f'Uncommitted changes: {", ".join(changes_list)}')

            # Recent commits
            commits = self.get_recent_commits(1)
            if commits:
                context_parts.append(f'Latest commit: {commits[0]}')

            if context_parts:
                return '\n'.join(context_parts)
        except Exception as e:
            logger.warning(f'Failed to get Git context: {e}')

        return None
