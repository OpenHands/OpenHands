import logging
import shlex
import subprocess
from pathlib import Path

from openhands.sdk.git.exceptions import GitCommandError, GitRepositoryError


logger = logging.getLogger(__name__)

# Git empty tree hash - this is a well-known constant in git
# representing the hash of an empty tree object
GIT_EMPTY_TREE_HASH = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def run_git_command(args: list[str], cwd: str | Path) -> str:
    """Run a git command safely without shell injection vulnerabilities.

    Args:
        args: List of command arguments (e.g., ['git', 'status', '--porcelain'])
        cwd: Working directory to run the command in

    Returns:
        Command output as string

    Raises:
        GitCommandError: If the git command fails
    """
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,  # Prevent hanging commands
        )

        if result.returncode != 0:
            cmd_str = shlex.join(args)
            error_msg = f"Git command failed: {cmd_str}"
            logger.error(
                f"{error_msg}. Exit code: {result.returncode}. Stderr: {result.stderr}"
            )
            raise GitCommandError(
                message=error_msg,
                command=args,
                exit_code=result.returncode,
                stderr=result.stderr.strip(),
            )

        logger.debug(f"Git command succeeded: {shlex.join(args)}")
        return result.stdout.strip()

    except subprocess.TimeoutExpired as e:
        cmd_str = shlex.join(args)
        error_msg = f"Git command timed out: {cmd_str}"
        logger.error(error_msg)
        raise GitCommandError(
            message=error_msg,
            command=args,
            exit_code=-1,
            stderr="Command timed out",
        ) from e
    except FileNotFoundError as e:
        error_msg = "Git command not found. Is git installed?"
        logger.error(error_msg)
        raise GitCommandError(
            message=error_msg,
            command=args,
            exit_code=-1,
            stderr="Git executable not found",
        ) from e


def get_valid_ref(repo_dir: str | Path) -> str | None:
    """Get a valid git reference to compare against.

    Tries multiple strategies to find a valid reference:
    1. Current branch's origin (e.g., origin/main)
    2. Default branch (e.g., origin/main, origin/master)
    3. Merge base with default branch
    4. Empty tree (for new repositories)

    Args:
        repo_dir: Path to the git repository

    Returns:
        Valid git reference hash, or None if no valid reference found
    """
    refs_to_try = []

    # Try current branch's origin
    try:
        current_branch = run_git_command(
            ["git", "--no-pager", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir
        )
        if current_branch and current_branch != "HEAD":  # Not in detached HEAD state
            refs_to_try.append(f"origin/{current_branch}")
            logger.debug(f"Added current branch reference: origin/{current_branch}")
    except GitCommandError:
        logger.debug("Could not get current branch name")

    # Try to get default branch from remote
    try:
        remote_info = run_git_command(
            ["git", "--no-pager", "remote", "show", "origin"], repo_dir
        )
        for line in remote_info.splitlines():
            if "HEAD branch:" in line:
                default_branch = line.split(":")[-1].strip()
                if default_branch:
                    refs_to_try.append(f"origin/{default_branch}")
                    logger.debug(
                        f"Added default branch reference: origin/{default_branch}"
                    )

                    # Also try merge base with default branch
                    try:
                        merge_base = run_git_command(
                            [
                                "git",
                                "--no-pager",
                                "merge-base",
                                "HEAD",
                                f"origin/{default_branch}",
                            ],
                            repo_dir,
                        )
                        if merge_base:
                            refs_to_try.append(merge_base)
                            logger.debug(f"Added merge base reference: {merge_base}")
                    except GitCommandError:
                        logger.debug("Could not get merge base")
                break
    except GitCommandError:
        logger.debug("Could not get remote information")

    # Add empty tree as fallback for new repositories
    refs_to_try.append(GIT_EMPTY_TREE_HASH)
    logger.debug(f"Added empty tree reference: {GIT_EMPTY_TREE_HASH}")

    # Find the first valid reference
    for ref in refs_to_try:
        try:
            result = run_git_command(
                ["git", "--no-pager", "rev-parse", "--verify", ref], repo_dir
            )
            if result:
                logger.debug(f"Using valid reference: {ref} -> {result}")
                return result
        except GitCommandError:
            logger.debug(f"Reference not valid: {ref}")
            continue

    logger.warning("No valid git reference found")
    return None


def validate_git_repository(repo_dir: str | Path) -> Path:
    """Validate that the given directory is a git repository.

    Args:
        repo_dir: Path to check

    Returns:
        Validated Path object

    Raises:
        GitRepositoryError: If not a valid git repository
    """
    repo_path = Path(repo_dir).resolve()

    if not repo_path.exists():
        raise GitRepositoryError(f"Directory does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise GitRepositoryError(f"Path is not a directory: {repo_path}")

    # Check if it's a git repository by looking for .git directory or file
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        # Maybe we're in a subdirectory, try to find the git root
        try:
            run_git_command(["git", "rev-parse", "--git-dir"], repo_path)
        except GitCommandError as e:
            raise GitRepositoryError(f"Not a git repository: {repo_path}") from e

    return repo_path
