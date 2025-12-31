"""Utilities for loading skills for V1 conversations.

This module provides functions to load skills from various sources:
- Global skills from OpenHands/skills/
- User skills from ~/.openhands/skills/
- Repository-level skills from the workspace

All skills are used in V1 conversations.
"""

import logging
import os
from pathlib import Path

import openhands
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderType
from openhands.integrations.service_types import AuthenticationError
from openhands.sdk.context.skills import Skill
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace

_logger = logging.getLogger(__name__)

# Path to global skills directory
GLOBAL_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(openhands.__file__)),
    'skills',
)
WORK_HOSTS_SKILL = """The user has access to the following hosts for accessing a web application,
each of which has a corresponding port:"""


def _find_and_load_global_skill_files(skill_dir: Path) -> list[Skill]:
    """Find and load all .md files from the global skills directory.

    Args:
        skill_dir: Path to the global skills directory

    Returns:
        List of Skill objects loaded from the files (excluding README.md)
    """
    skills = []

    try:
        # Find all .md files in the directory (excluding README.md)
        md_files = [f for f in skill_dir.glob('*.md') if f.name.lower() != 'readme.md']

        # Load skills from the found files
        for file_path in md_files:
            try:
                skill = Skill.load(file_path, skill_dir)
                skills.append(skill)
                _logger.debug(f'Loaded global skill: {skill.name} from {file_path}')
            except Exception as e:
                _logger.warning(
                    f'Failed to load global skill from {file_path}: {str(e)}'
                )

    except Exception as e:
        _logger.debug(f'Failed to find global skill files: {str(e)}')

    return skills


def load_sandbox_skills(sandbox: SandboxInfo) -> list[Skill]:
    """Load skills specific to the sandbox, including exposed ports / urls."""
    if not sandbox.exposed_urls:
        return []
    urls = [url for url in sandbox.exposed_urls if url.name.startswith('WORKER_')]
    if not urls:
        return []
    content_list = [WORK_HOSTS_SKILL]
    for url in urls:
        content_list.append(f'* {url.url} (port {url.port})')
    content = '\n'.join(content_list)
    return [Skill(name='work_hosts', content=content, trigger=None)]


def load_global_skills() -> list[Skill]:
    """Load global skills from OpenHands/skills/ directory.

    Returns:
        List of Skill objects loaded from global skills directory.
        Returns empty list if directory doesn't exist or on errors.
    """
    skill_dir = Path(GLOBAL_SKILLS_DIR)

    # Check if directory exists
    if not skill_dir.exists():
        _logger.debug(f'Global skills directory does not exist: {skill_dir}')
        return []

    try:
        _logger.info(f'Loading global skills from {skill_dir}')

        # Find and load all .md files from the directory
        skills = _find_and_load_global_skill_files(skill_dir)

        _logger.info(f'Loaded {len(skills)} global skills: {[s.name for s in skills]}')

        return skills

    except Exception as e:
        _logger.warning(f'Failed to load global skills: {str(e)}')
        return []


def _determine_repo_root(working_dir: str, selected_repository: str | None) -> str:
    """Determine the repository root directory.

    Args:
        working_dir: Base working directory path
        selected_repository: Repository name (e.g., 'owner/repo') or None

    Returns:
        Path to the repository root directory
    """
    if selected_repository:
        repo_name = selected_repository.split('/')[-1]
        return f'{working_dir}/{repo_name}'
    return working_dir


async def _is_gitlab_repository(repo_name: str, user_context: UserContext) -> bool:
    """Check if a repository is hosted on GitLab.

    Args:
        repo_name: Repository name (e.g., "gitlab.com/org/repo" or "org/repo")
        user_context: UserContext to access provider handler

    Returns:
        True if the repository is hosted on GitLab, False otherwise
    """
    try:
        provider_handler = await user_context.get_provider_handler()  # type: ignore[attr-defined]
        repository = await provider_handler.verify_repo_provider(repo_name)
        return repository.git_provider == ProviderType.GITLAB
    except Exception:
        # If we can't determine the provider, assume it's not GitLab
        # This is a safe fallback since we'll just use the default .openhands
        return False


async def _is_azure_devops_repository(
    repo_name: str, user_context: UserContext
) -> bool:
    """Check if a repository is hosted on Azure DevOps.

    Args:
        repo_name: Repository name (e.g., "org/project/repo")
        user_context: UserContext to access provider handler

    Returns:
        True if the repository is hosted on Azure DevOps, False otherwise
    """
    try:
        provider_handler = await user_context.get_provider_handler()  # type: ignore[attr-defined]
        repository = await provider_handler.verify_repo_provider(repo_name)
        return repository.git_provider == ProviderType.AZURE_DEVOPS
    except Exception:
        # If we can't determine the provider, assume it's not Azure DevOps
        return False


async def _determine_org_repo_path(
    selected_repository: str, user_context: UserContext
) -> tuple[str, str]:
    """Determine the organization repository path and organization name.

    Args:
        selected_repository: Repository name (e.g., 'owner/repo' or 'org/project/repo')
        user_context: UserContext to access provider handler

    Returns:
        Tuple of (org_repo_path, org_name) where:
        - org_repo_path: Full path to org-level config repo
        - org_name: Organization name extracted from repository

    Examples:
        - GitHub/Bitbucket: ('owner/.openhands', 'owner')
        - GitLab: ('owner/openhands-config', 'owner')
        - Azure DevOps: ('org/openhands-config/openhands-config', 'org')
    """
    repo_parts = selected_repository.split('/')

    # Determine repository type
    is_azure_devops = await _is_azure_devops_repository(
        selected_repository, user_context
    )
    is_gitlab = await _is_gitlab_repository(selected_repository, user_context)

    # Extract the org/user name
    # Azure DevOps format: org/project/repo (3 parts) - extract org (first part)
    # GitHub/GitLab/Bitbucket format: owner/repo (2 parts) - extract owner (first part)
    if is_azure_devops and len(repo_parts) >= 3:
        org_name = repo_parts[0]  # Get org from org/project/repo
    else:
        org_name = repo_parts[-2]  # Get owner from owner/repo

    # For GitLab and Azure DevOps, use openhands-config (since .openhands is not a valid repo name)
    # For other providers, use .openhands
    if is_gitlab:
        org_openhands_repo = f'{org_name}/openhands-config'
    elif is_azure_devops:
        # Azure DevOps format: org/project/repo
        # For org-level config, use: org/openhands-config/openhands-config
        org_openhands_repo = f'{org_name}/openhands-config/openhands-config'
    else:
        org_openhands_repo = f'{org_name}/.openhands'

    return org_openhands_repo, org_name


async def _read_file_from_workspace(
    workspace: AsyncRemoteWorkspace, file_path: str, working_dir: str
) -> str | None:
    """Read file content from remote workspace.

    Args:
        workspace: AsyncRemoteWorkspace to execute commands
        file_path: Path to the file to read
        working_dir: Working directory for command execution

    Returns:
        File content as string, or None if file doesn't exist or read fails
    """
    try:
        result = await workspace.execute_command(
            f'cat {file_path}', cwd=working_dir, timeout=10.0
        )
        if result.exit_code == 0 and result.stdout.strip():
            return result.stdout
        return None
    except Exception as e:
        _logger.debug(f'Failed to read file {file_path}: {str(e)}')
        return None


async def _load_special_files(
    workspace: AsyncRemoteWorkspace, repo_root: str, working_dir: str
) -> list[Skill]:
    """Load special skill files from repository root.

    Loads: .cursorrules, agents.md, agent.md

    Args:
        workspace: AsyncRemoteWorkspace to execute commands
        repo_root: Path to repository root directory
        working_dir: Working directory for command execution

    Returns:
        List of Skill objects loaded from special files
    """
    skills = []
    special_files = ['.cursorrules', 'agents.md', 'agent.md']

    for filename in special_files:
        file_path = f'{repo_root}/{filename}'
        content = await _read_file_from_workspace(workspace, file_path, working_dir)

        if content:
            try:
                # Use simple string path to avoid Path filesystem operations
                skill = Skill.load(path=filename, skill_dir=None, file_content=content)
                skills.append(skill)
                _logger.debug(f'Loaded special file skill: {skill.name}')
            except Exception as e:
                _logger.warning(f'Failed to create skill from {filename}: {str(e)}')

    return skills


async def _find_and_load_skill_md_files(
    workspace: AsyncRemoteWorkspace, skill_dir: str, working_dir: str
) -> list[Skill]:
    """Find and load all .md files from a skills directory in the workspace.

    Args:
        workspace: AsyncRemoteWorkspace to execute commands
        skill_dir: Path to skills directory
        working_dir: Working directory for command execution

    Returns:
        List of Skill objects loaded from the files (excluding README.md)
    """
    skills = []

    try:
        # Find all .md files in the directory
        result = await workspace.execute_command(
            f"find {skill_dir} -type f -name '*.md' 2>/dev/null || true",
            cwd=working_dir,
            timeout=10.0,
        )

        if result.exit_code == 0 and result.stdout.strip():
            file_paths = [
                f.strip()
                for f in result.stdout.strip().split('\n')
                if f.strip() and 'README.md' not in f
            ]

            # Load skills from the found files
            for file_path in file_paths:
                content = await _read_file_from_workspace(
                    workspace, file_path, working_dir
                )

                if content:
                    # Calculate relative path for skill name
                    rel_path = file_path.replace(f'{skill_dir}/', '')
                    try:
                        # Use simple string path to avoid Path filesystem operations
                        skill = Skill.load(
                            path=rel_path, skill_dir=None, file_content=content
                        )
                        skills.append(skill)
                        _logger.debug(f'Loaded repo skill: {skill.name}')
                    except Exception as e:
                        _logger.warning(
                            f'Failed to create skill from {rel_path}: {str(e)}'
                        )

    except Exception as e:
        _logger.debug(f'Failed to find skill files in {skill_dir}: {str(e)}')

    return skills


def _merge_repo_skills_with_precedence(
    special_skills: list[Skill],
    skills_dir_skills: list[Skill],
    microagents_dir_skills: list[Skill],
) -> list[Skill]:
    """Merge repository skills with precedence order.

    Precedence (highest to lowest):
    1. Special files (repo root)
    2. .openhands/skills/ directory
    3. .openhands/microagents/ directory (backward compatibility)

    Args:
        special_skills: Skills from special files in repo root
        skills_dir_skills: Skills from .openhands/skills/ directory
        microagents_dir_skills: Skills from .openhands/microagents/ directory

    Returns:
        Deduplicated list of skills with proper precedence
    """
    # Use a dict to deduplicate by name, with earlier sources taking precedence
    skills_by_name = {}
    for skill in special_skills + skills_dir_skills + microagents_dir_skills:
        # Only add if not already present (earlier sources win)
        if skill.name not in skills_by_name:
            skills_by_name[skill.name] = skill

    return list(skills_by_name.values())


async def load_repo_skills(
    workspace: AsyncRemoteWorkspace,
    selected_repository: str | None,
    working_dir: str,
) -> list[Skill]:
    """Load repository-level skills from the workspace.

    Loads skills from:
    1. Special files in repo root: .cursorrules, agents.md, agent.md
    2. .md files in .openhands/skills/ directory (preferred)
    3. .md files in .openhands/microagents/ directory (for backward compatibility)

    Args:
        workspace: AsyncRemoteWorkspace to execute commands in the sandbox
        selected_repository: Repository name (e.g., 'owner/repo') or None
        working_dir: Working directory path

    Returns:
        List of Skill objects loaded from repository.
        Returns empty list on errors.
    """
    try:
        # Determine repository root directory
        repo_root = _determine_repo_root(working_dir, selected_repository)
        _logger.info(f'Loading repo skills from {repo_root}')

        # Load special files from repo root
        special_skills = await _load_special_files(workspace, repo_root, working_dir)

        # Load .md files from .openhands/skills/ directory (preferred)
        skills_dir = f'{repo_root}/.openhands/skills'
        skills_dir_skills = await _find_and_load_skill_md_files(
            workspace, skills_dir, working_dir
        )

        # Load .md files from .openhands/microagents/ directory (backward compatibility)
        microagents_dir = f'{repo_root}/.openhands/microagents'
        microagents_dir_skills = await _find_and_load_skill_md_files(
            workspace, microagents_dir, working_dir
        )

        # Merge all loaded skills with proper precedence
        all_skills = _merge_repo_skills_with_precedence(
            special_skills, skills_dir_skills, microagents_dir_skills
        )

        _logger.info(
            f'Loaded {len(all_skills)} repo skills: {[s.name for s in all_skills]}'
        )

        return all_skills

    except Exception as e:
        _logger.warning(f'Failed to load repo skills: {str(e)}')
        return []


def _validate_repository_for_org_skills(selected_repository: str) -> bool:
    """Validate that the repository path has sufficient parts for org skills.

    Args:
        selected_repository: Repository name (e.g., 'owner/repo')

    Returns:
        True if repository is valid for org skills loading, False otherwise
    """
    repo_parts = selected_repository.split('/')
    if len(repo_parts) < 2:
        _logger.warning(
            f'Repository path has insufficient parts ({len(repo_parts)} < 2), skipping org-level skills'
        )
        return False
    return True


async def _get_org_repository_url(
    org_openhands_repo: str, user_context: UserContext
) -> str | None:
    """Get authenticated Git URL for organization repository.

    Args:
        org_openhands_repo: Organization repository path
        user_context: UserContext to access authentication

    Returns:
        Authenticated Git URL if successful, None otherwise
    """
    try:
        remote_url = await user_context.get_authenticated_git_url(org_openhands_repo)
        return remote_url
    except AuthenticationError as e:
        _logger.debug(
            f'org-level skill directory {org_openhands_repo} not found: {str(e)}'
        )
        return None
    except Exception as e:
        _logger.debug(
            f'Failed to get authenticated URL for {org_openhands_repo}: {str(e)}'
        )
        return None


async def _clone_org_repository(
    workspace: AsyncRemoteWorkspace,
    remote_url: str,
    org_repo_dir: str,
    working_dir: str,
    org_openhands_repo: str,
) -> bool:
    """Clone organization repository to temporary directory.

    Args:
        workspace: AsyncRemoteWorkspace to execute commands
        remote_url: Authenticated Git URL
        org_repo_dir: Temporary directory path for cloning
        working_dir: Working directory for command execution
        org_openhands_repo: Organization repository path (for logging)

    Returns:
        True if clone successful, False otherwise
    """
    _logger.debug(f'Creating temporary directory for org repo: {org_repo_dir}')

    # Clone the repo (shallow clone for efficiency)
    clone_cmd = f'GIT_TERMINAL_PROMPT=0 git clone --depth 1 {remote_url} {org_repo_dir}'
    _logger.info('Executing clone command for org-level repo')

    result = await workspace.execute_command(clone_cmd, working_dir, timeout=120.0)

    if result.exit_code != 0:
        _logger.info(
            f'No org-level skills found at {org_openhands_repo} (exit_code: {result.exit_code})'
        )
        _logger.debug(f'Clone command output: {result.stderr}')
        return False

    _logger.info(f'Successfully cloned org-level skills from {org_openhands_repo}')
    return True


async def _load_skills_from_org_directories(
    workspace: AsyncRemoteWorkspace, org_repo_dir: str, working_dir: str
) -> tuple[list[Skill], list[Skill]]:
    """Load skills from both skills/ and microagents/ directories in org repo.

    Args:
        workspace: AsyncRemoteWorkspace to execute commands
        org_repo_dir: Path to cloned organization repository
        working_dir: Working directory for command execution

    Returns:
        Tuple of (skills_dir_skills, microagents_dir_skills)
    """
    skills_dir = f'{org_repo_dir}/skills'
    skills_dir_skills = await _find_and_load_skill_md_files(
        workspace, skills_dir, working_dir
    )

    microagents_dir = f'{org_repo_dir}/microagents'
    microagents_dir_skills = await _find_and_load_skill_md_files(
        workspace, microagents_dir, working_dir
    )

    return skills_dir_skills, microagents_dir_skills


def _merge_org_skills_with_precedence(
    skills_dir_skills: list[Skill], microagents_dir_skills: list[Skill]
) -> list[Skill]:
    """Merge skills from skills/ and microagents/ with proper precedence.

    Precedence: skills/ > microagents/ (skills/ overrides microagents/ for same name)

    Args:
        skills_dir_skills: Skills loaded from skills/ directory
        microagents_dir_skills: Skills loaded from microagents/ directory

    Returns:
        Merged list of skills with proper precedence applied
    """
    skills_by_name = {}
    for skill in microagents_dir_skills + skills_dir_skills:
        # Later sources (skills/) override earlier ones (microagents/)
        if skill.name not in skills_by_name:
            skills_by_name[skill.name] = skill
        else:
            _logger.debug(
                f'Overriding org skill "{skill.name}" from microagents/ with skills/'
            )
            skills_by_name[skill.name] = skill

    return list(skills_by_name.values())


async def _cleanup_org_repository(
    workspace: AsyncRemoteWorkspace, org_repo_dir: str, working_dir: str
) -> None:
    """Clean up cloned organization repository directory.

    Args:
        workspace: AsyncRemoteWorkspace to execute commands
        org_repo_dir: Path to cloned organization repository
        working_dir: Working directory for command execution
    """
    cleanup_cmd = f'rm -rf {org_repo_dir}'
    await workspace.execute_command(cleanup_cmd, working_dir, timeout=10.0)


async def load_org_skills(
    workspace: AsyncRemoteWorkspace,
    selected_repository: str | None,
    working_dir: str,
    user_context: UserContext,
) -> list[Skill]:
    """Load organization-level skills from the organization repository.

    For example, if the repository is github.com/acme-co/api, this will check if
    github.com/acme-co/.openhands exists. If it does, it will clone it and load
    the skills from both the ./skills/ and ./microagents/ folders.

    For GitLab repositories, it will use openhands-config instead of .openhands
    since GitLab doesn't support repository names starting with non-alphanumeric
    characters.

    For Azure DevOps repositories, it will use org/openhands-config/openhands-config
    format to match Azure DevOps's three-part repository structure (org/project/repo).

    Args:
        workspace: AsyncRemoteWorkspace to execute commands in the sandbox
        selected_repository: Repository name (e.g., 'owner/repo') or None
        working_dir: Working directory path
        user_context: UserContext to access provider handler and authentication

    Returns:
        List of Skill objects loaded from organization repository.
        Returns empty list if no repository selected or on errors.
    """
    if not selected_repository:
        return []

    try:
        _logger.debug(
            f'Starting org-level skill loading for repository: {selected_repository}'
        )

        # Validate repository path
        if not _validate_repository_for_org_skills(selected_repository):
            return []

        # Determine organization repository path
        org_openhands_repo, org_name = await _determine_org_repo_path(
            selected_repository, user_context
        )

        _logger.info(f'Checking for org-level skills at {org_openhands_repo}')

        # Get authenticated URL for org repository
        remote_url = await _get_org_repository_url(org_openhands_repo, user_context)
        if not remote_url:
            return []

        # Clone the organization repository
        org_repo_dir = f'{working_dir}/_org_openhands_{org_name}'
        clone_success = await _clone_org_repository(
            workspace, remote_url, org_repo_dir, working_dir, org_openhands_repo
        )
        if not clone_success:
            return []

        # Load skills from both skills/ and microagents/ directories
        (
            skills_dir_skills,
            microagents_dir_skills,
        ) = await _load_skills_from_org_directories(
            workspace, org_repo_dir, working_dir
        )

        # Merge skills with proper precedence
        loaded_skills = _merge_org_skills_with_precedence(
            skills_dir_skills, microagents_dir_skills
        )

        _logger.info(
            f'Loaded {len(loaded_skills)} skills from org-level repository {org_openhands_repo}: {[s.name for s in loaded_skills]}'
        )

        # Clean up the org repo directory
        await _cleanup_org_repository(workspace, org_repo_dir, working_dir)

        return loaded_skills

    except AuthenticationError as e:
        _logger.debug(f'org-level skill directory not found: {str(e)}')
        return []
    except Exception as e:
        _logger.warning(f'Failed to load org-level skills: {str(e)}')
        return []


def merge_skills(skill_lists: list[list[Skill]]) -> list[Skill]:
    """Merge multiple skill lists, avoiding duplicates by name.

    Later lists take precedence over earlier lists for duplicate names.

    Args:
        skill_lists: List of skill lists to merge

    Returns:
        Deduplicated list of skills with later lists overriding earlier ones
    """
    skills_by_name = {}

    for skill_list in skill_lists:
        for skill in skill_list:
            if skill.name in skills_by_name:
                _logger.debug(
                    f'Overriding skill "{skill.name}" from earlier source with later source'
                )
            skills_by_name[skill.name] = skill

    result = list(skills_by_name.values())
    _logger.debug(f'Merged skills: {[s.name for s in result]}')
    return result
