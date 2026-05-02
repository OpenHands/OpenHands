"""Service for resolving and cloning personal skills repositories.

Handles resolving a repo URL to a commit hash and cloning the repo
at a pinned commit for skill loading.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse, urlunparse

_logger = logging.getLogger(__name__)

PERSONAL_SKILLS_CACHE_DIR = Path.home() / '.openhands' / 'personal_skills_repo'


def _validate_git_url(url: str) -> None:
    """Validate that the URL is a safe Git repository URL.

    Raises:
        ValueError: If the URL is not a valid HTTPS git URL.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ('https', 'http'):
        raise ValueError(f'Unsupported URL scheme: {parsed.scheme or "none"}')
    if not parsed.netloc:
        raise ValueError('Invalid Git URL: missing hostname')


def _normalize_clone_url(url: str) -> str:
    """Ensure URL ends with .git for cloning."""
    url = url.strip().rstrip('/')
    return url if url.endswith('.git') else url + '.git'


def _inject_token(clone_url: str, token: str | None) -> str:
    """Inject an auth token into an HTTPS git URL for private repo access."""
    if not token:
        return clone_url
    parsed = urlparse(clone_url)
    if parsed.scheme != 'https':
        return clone_url
    netloc = f'x-access-token:{token}@{parsed.netloc}'
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def resolve_repo_commit(repo_url: str, token: str | None = None) -> str:
    """Resolve a git repo URL to its current HEAD commit hash.

    Args:
        repo_url: Git repository URL (HTTPS).
        token: Optional auth token for private repos.

    Returns:
        The full SHA-1 commit hash of HEAD.

    Raises:
        ValueError: If the repo URL cannot be resolved.
    """
    normalized = _normalize_clone_url(repo_url)
    _validate_git_url(normalized)
    clone_url = _inject_token(normalized, token)
    try:
        result = subprocess.run(
            ['git', 'ls-remote', clone_url, 'HEAD'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ValueError(f'Failed to resolve repo: {result.stderr.strip()}')
        output = result.stdout.strip()
        if not output:
            raise ValueError(f'No HEAD ref found for repo {repo_url}')
        return output.split()[0]
    except subprocess.TimeoutExpired:
        raise ValueError(f'Timeout resolving repo {repo_url}')
    except FileNotFoundError:
        raise ValueError('git is not installed or not in PATH')


def clone_repo_at_commit(repo_url: str, commit: str, token: str | None = None) -> Path:
    """Clone a repo at a specific commit into the personal skills cache.

    If the cache already has the correct commit checked out, this is a no-op.

    Args:
        repo_url: Git repository URL.
        commit: Full commit hash to checkout.
        token: Optional auth token for private repos.

    Returns:
        Path to the cloned repo directory.

    Raises:
        ValueError: If cloning or checkout fails.
    """
    normalized = _normalize_clone_url(repo_url)
    _validate_git_url(normalized)
    cache_dir = PERSONAL_SKILLS_CACHE_DIR
    clone_url = _inject_token(normalized, token)

    # Check if already at the right commit
    if cache_dir.exists():
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=cache_dir,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip() == commit:
                return cache_dir
        except Exception:
            pass
        shutil.rmtree(cache_dir, ignore_errors=True)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ['git', 'clone', '--no-checkout', clone_url, str(cache_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ValueError(f'Clone failed: {result.stderr.strip()}')
        result = subprocess.run(
            ['git', 'checkout', commit],
            capture_output=True,
            text=True,
            cwd=cache_dir,
            timeout=30,
        )
        if result.returncode != 0:
            raise ValueError(f'Checkout failed: {result.stderr.strip()}')
        return cache_dir
    except subprocess.TimeoutExpired:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise ValueError(f'Timeout cloning repo {repo_url}')
    except Exception as e:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise ValueError(f'Failed to clone repo {repo_url}: {e}')


def get_skills_dir_from_repo(repo_dir: Path) -> Path | None:
    """Find the skills/microagents directory inside a cloned repo.

    Looks for common conventions:
    - .openhands/microagents/
    - skills/
    - .agents/skills/
    - Root if it contains .md files

    Returns:
        Path to the skills directory, or None if not found.
    """
    candidates = [
        repo_dir / '.openhands' / 'microagents',
        repo_dir / 'skills',
        repo_dir / '.agents' / 'skills',
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    md_files = list(repo_dir.glob('*.md'))
    if md_files:
        return repo_dir
    return None


def load_personal_repo_skills() -> list[tuple[str, str, list[str]]]:
    """Load skills from the personal skills repo cache.

    Returns:
        List of (name, content, triggers) tuples for each skill found.
        Returns empty list if no repo is configured or cache doesn't exist.
    """
    if not PERSONAL_SKILLS_CACHE_DIR.exists():
        return []

    skills_dir = get_skills_dir_from_repo(PERSONAL_SKILLS_CACHE_DIR)
    if not skills_dir:
        return []

    import io

    import frontmatter

    results = []
    for md_file in skills_dir.rglob('*.md'):
        if md_file.name == 'README.md':
            continue
        try:
            text = md_file.read_text(encoding='utf-8')
            loaded = frontmatter.load(io.StringIO(text))
            fm = loaded.metadata or {}
            name = fm.get('name') or md_file.stem
            content = loaded.content
            triggers = fm.get('triggers') or []
            results.append((name, content, triggers))
        except Exception as e:
            _logger.debug(f'Failed to load skill from {md_file}: {e}')
            continue
    return results
