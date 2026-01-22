"""Utilities for loading skills for V1 conversations.

This module provides functions to load skills from the agent-server,
which centralizes all skill loading logic. The app-server acts as a
thin proxy that:
1. Builds the org_config with authentication information
2. Builds the sandbox_config with exposed URLs
3. Calls the agent-server's /api/skills endpoint

All source-specific skill loading is handled by the agent-server.
"""

import logging

import httpx
from pydantic import BaseModel

from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.user.user_context import UserContext
from openhands.integrations.provider import ProviderType
from openhands.integrations.service_types import AuthenticationError
from openhands.sdk.context.skills import Skill
from openhands.sdk.context.skills.trigger import KeywordTrigger, TaskTrigger

_logger = logging.getLogger(__name__)


class ExposedUrlConfig(BaseModel):
    """Configuration for an exposed URL in sandbox config."""

    name: str
    url: str
    port: int


WORK_HOSTS_SKILL_FOOTER = """
When starting a web server, use the corresponding ports via environment variables:
- $WORKER_1 for the first port
- $WORKER_2 for the second port

**CRITICAL: You MUST enable CORS and bind to 0.0.0.0.** Without CORS headers, the App tab cannot detect your server and will show an empty state.

Example (Flask):
```python
from flask_cors import CORS
CORS(app)
app.run(host='0.0.0.0', port=int(os.environ.get('WORKER_1', 12000)))
```"""


class SandboxConfig(BaseModel):
    """Sandbox configuration for agent-server API request."""

    exposed_urls: list[ExposedUrlConfig]


class OrgConfig(BaseModel):
    """Organization configuration for agent-server API request."""

    repository: str
    provider: str
    org_repo_url: str
    org_name: str


class SkillInfo(BaseModel):
    """Skill information from agent-server API response."""

    name: str
    content: str
    triggers: list[str] = []
    source: str | None = None
    description: str | None = None
    is_agentskills_format: bool = False


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
        repository = await provider_handler.verify_repo_provider(
            repo_name, is_optional=True
        )
        return repository.git_provider == ProviderType.GITLAB
    except Exception:
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
        repository = await provider_handler.verify_repo_provider(
            repo_name, is_optional=True
        )
        return repository.git_provider == ProviderType.AZURE_DEVOPS
    except Exception:
        return False


async def _get_provider_type(
    selected_repository: str, user_context: UserContext
) -> str:
    """Determine the Git provider type for a repository.

    Args:
        selected_repository: Repository name (e.g., 'owner/repo')
        user_context: UserContext to access provider handler

    Returns:
        Provider type string: 'github', 'gitlab', 'azure', or 'bitbucket'
    """
    is_gitlab = await _is_gitlab_repository(selected_repository, user_context)
    if is_gitlab:
        return 'gitlab'

    is_azure = await _is_azure_devops_repository(selected_repository, user_context)
    if is_azure:
        return 'azure'

    # Default to github (covers github and bitbucket)
    return 'github'


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

    is_azure_devops = await _is_azure_devops_repository(
        selected_repository, user_context
    )
    is_gitlab = await _is_gitlab_repository(selected_repository, user_context)

    if is_azure_devops and len(repo_parts) >= 3:
        org_name = repo_parts[0]
    else:
        org_name = repo_parts[-2]

    if is_gitlab:
        org_openhands_repo = f'{org_name}/openhands-config'
    elif is_azure_devops:
        org_openhands_repo = f'{org_name}/openhands-config/openhands-config'
    else:
        org_openhands_repo = f'{org_name}/.openhands'

    return org_openhands_repo, org_name


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
        remote_url = await user_context.get_authenticated_git_url(
            org_openhands_repo, is_optional=True
        )
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


async def build_org_config(
    selected_repository: str | None,
    user_context: UserContext,
) -> OrgConfig | None:
    """Build organization config for agent-server API request.

    Args:
        selected_repository: Repository name (e.g., 'owner/repo') or None
        user_context: UserContext to access authentication and provider info

    Returns:
        org_config dict if org repository exists and is accessible, None otherwise
    """
    if not selected_repository:
        return None

    repo_parts = selected_repository.split('/')
    if len(repo_parts) < 2:
        _logger.warning(
            f'Repository path has insufficient parts ({len(repo_parts)} < 2), '
            f'skipping org-level skills'
        )
        return None

    try:
        org_openhands_repo, org_name = await _determine_org_repo_path(
            selected_repository, user_context
        )

        org_repo_url = await _get_org_repository_url(org_openhands_repo, user_context)
        if not org_repo_url:
            return None

        provider = await _get_provider_type(selected_repository, user_context)

        return OrgConfig(
            repository=selected_repository,
            provider=provider,
            org_repo_url=org_repo_url,
            org_name=org_name,
        )

    except Exception as e:
        _logger.debug(f'Failed to build org config: {str(e)}')
        return None


def build_sandbox_config(sandbox: SandboxInfo) -> SandboxConfig | None:
    """Build sandbox config for agent-server API request.

    Args:
        sandbox: SandboxInfo containing exposed URLs

    Returns:
        sandbox_config dict if there are exposed URLs, None otherwise
    """
    if not sandbox.exposed_urls:
        return None

    exposed_urls = [
        ExposedUrlConfig(name=url.name, url=url.url, port=url.port)
        for url in sandbox.exposed_urls
    ]

    return SandboxConfig(exposed_urls=exposed_urls)


async def load_skills_from_agent_server(
    agent_server_url: str,
    session_api_key: str | None,
    project_dir: str,
    org_config: OrgConfig | None = None,
    sandbox_config: SandboxConfig | None = None,
    load_public: bool = True,
    load_user: bool = True,
    load_project: bool = True,
    load_org: bool = True,
) -> list[Skill]:
    """Load all skills from the agent-server.

    This function makes a single API call to the agent-server's /api/skills
    endpoint to load and merge skills from all configured sources.

    Args:
        agent_server_url: URL of the agent server (e.g., 'http://localhost:8000')
        session_api_key: Session API key for authentication (optional)
        project_dir: Workspace directory path for project skills
        org_config: Organization skills configuration (optional)
        sandbox_config: Sandbox skills configuration (optional)
        load_public: Whether to load public skills (default: True)
        load_user: Whether to load user skills (default: True)
        load_project: Whether to load project skills (default: True)
        load_org: Whether to load organization skills (default: True)

    Returns:
        List of Skill objects merged from all sources.
        Returns empty list on error.
    """
    try:
        # Build request payload
        payload = {
            'load_public': load_public,
            'load_user': load_user,
            'load_project': load_project,
            'load_org': load_org,
            'project_dir': project_dir,
            'org_config': org_config.model_dump() if org_config else None,
            'sandbox_config': sandbox_config.model_dump() if sandbox_config else None,
        }

        # Build headers
        headers = {'Content-Type': 'application/json'}
        if session_api_key:
            headers['X-Session-API-Key'] = session_api_key

        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{agent_server_url}/api/skills',
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()

            data = response.json()

        # Convert response to Skill objects
        skills: list[Skill] = []
        for skill_data_dict in data.get('skills', []):
            try:
                skill_info = SkillInfo.model_validate(skill_data_dict)
                skill = _convert_skill_info_to_skill(skill_info)
                skills.append(skill)
            except Exception as e:
                skill_name = (
                    skill_data_dict.get('name', 'unknown')
                    if isinstance(skill_data_dict, dict)
                    else 'unknown'
                )
                _logger.warning(f'Failed to convert skill {skill_name}: {e}')

        sources = data.get('sources', {})
        _logger.info(
            f'Loaded {len(skills)} skills from agent-server: '
            f'sources={sources}, names={[s.name for s in skills]}'
        )

        return skills

    except httpx.HTTPStatusError as e:
        _logger.warning(
            f'Agent-server returned error status {e.response.status_code}: '
            f'{e.response.text}'
        )
        return []
    except httpx.RequestError as e:
        _logger.warning(f'Failed to connect to agent-server: {e}')
        return []
    except Exception as e:
        _logger.warning(f'Failed to load skills from agent-server: {e}')
        return []


def _convert_skill_info_to_skill(skill_info: SkillInfo) -> Skill:
    """Convert skill info from API response to Skill object.

    Args:
        skill_info: SkillInfo model from API response

    Returns:
        Skill object
    """
    trigger = None

    if skill_info.triggers:
        # Determine trigger type based on content
        if any(t.startswith('/') for t in skill_info.triggers):
            trigger = TaskTrigger(triggers=skill_info.triggers)
        else:
            trigger = KeywordTrigger(keywords=skill_info.triggers)

    return Skill(
        name=skill_info.name,
        content=skill_info.content,
        trigger=trigger,
        source=skill_info.source,
        description=skill_info.description,
        is_agentskills_format=skill_info.is_agentskills_format,
    )
