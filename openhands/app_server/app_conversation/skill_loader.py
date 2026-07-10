"""Utilities for loading skills for V1 conversations.

This module provides functions to load skills from the agent-server,
which centralizes all skill loading logic. The app-server acts as a
thin proxy that:
1. Builds the org_config with authentication information
2. Builds the sandbox_config with exposed URLs
3. Calls the agent-server's /api/skills endpoint

All source-specific skill loading is handled by the agent-server.
"""

import asyncio
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from openhands.app_server.integrations.provider import ProviderHandler, ProviderType
from openhands.app_server.integrations.service_types import AuthenticationError
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.settings.settings_models import MarketplaceRegistration
from openhands.app_server.user.user_context import UserContext
from openhands.sdk.skills import KeywordTrigger, Skill, TaskTrigger

_logger = logging.getLogger(__name__)


class ExposedUrlConfig(BaseModel):
    """Configuration for an exposed URL in sandbox config."""

    name: str
    url: str
    port: int


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
        provider_handler = await user_context.get_provider_handler()
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
        provider_handler = await user_context.get_provider_handler()
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


def _candidate_repo_paths(provider: ProviderType, owner: str) -> list[str]:
    """Return the global skill-repo paths for an owner, by provider convention.

    GitHub-style providers expose two independent repos (``.openhands`` and
    ``.agents``) that are loaded concurrently. GitLab and Azure DevOps use a
    single ``openhands-config`` repository (they have no ``.agents`` analog).

    Args:
        provider: Git provider the owner belongs to.
        owner: Account login or organization/group name.

    Returns:
        List of repository paths (e.g., ['owner/.openhands', 'owner/.agents']).
    """
    if provider == ProviderType.GITLAB:
        return [f'{owner}/openhands-config']
    if provider == ProviderType.AZURE_DEVOPS:
        return [f'{owner}/openhands-config/openhands-config']
    return [f'{owner}/.openhands', f'{owner}/.agents']


async def _enumerate_owners_for_provider(
    provider_handler: ProviderHandler, provider: ProviderType
) -> list[str]:
    """Collect skill-repo owners for a provider: the user's login plus their orgs.

    Failures are swallowed so that a single provider error never prevents skill
    loading; the underlying enumeration helpers already return ``[]`` on error.

    Args:
        provider_handler: Handler holding the user's provider tokens.
        provider: Provider to enumerate owners for.

    Returns:
        List of owner names (login first, then organizations/groups).
    """
    owners: list[str] = []

    try:
        service = provider_handler.get_service(provider)
        user = await service.get_user()
        if user and user.login:
            owners.append(user.login)
    except Exception as e:
        _logger.debug(f'Failed to get user login for provider {provider}: {e}')

    try:
        if provider == ProviderType.GITHUB:
            owners.extend(await provider_handler.get_github_organizations())
        elif provider == ProviderType.GITLAB:
            owners.extend(await provider_handler.get_gitlab_groups())
        elif provider == ProviderType.BITBUCKET:
            owners.extend(await provider_handler.get_bitbucket_workspaces())
        elif provider == ProviderType.AZURE_DEVOPS:
            owners.extend(await provider_handler.get_azure_devops_organizations())
        # Bitbucket Data Center is intentionally excluded: get_bitbucket_dc_projects
        # hits /rest/api/1.0/projects, which returns every project the user can
        # *browse* on the server (not their memberships). Enumerating it here would
        # fan out to ~all projects on the instance and inject their skills into every
        # conversation. BBDC org skills still load via the selected-repo path.
    except Exception as e:
        _logger.debug(f'Failed to enumerate orgs for provider {provider}: {e}')

    return owners


# Upper bound on how many global skill repos we will verify for a single
# conversation, and how many of those verifications run concurrently. These
# cap the HTTP fan-out against the user's git provider(s) on conversation start.
_MAX_ORG_CANDIDATES = 30
_URL_RESOLVE_CONCURRENCY = 8


async def build_org_configs(
    selected_repository: str | None,
    user_context: UserContext,
) -> list[OrgConfig]:
    """Build the list of global org/user skill-repo configs for the agent-server.

    Skills are loaded for every conversation regardless of repository selection.
    The list always covers the authenticated user's account and every
    organization/group they belong to (across all authenticated providers),
    resolving both ``.openhands`` and ``.agents`` repos for GitHub-style
    providers. When a repository is selected, the repo owner's repos are
    included too. Only repos whose authenticated URL resolves are returned, and
    duplicate repo paths are collapsed. Failures degrade to fewer (or no)
    configs rather than raising.

    Args:
        selected_repository: Repository name (e.g., 'owner/repo') or None
        user_context: UserContext to access authentication and provider info

    Returns:
        List of OrgConfig for every accessible global skill repository.
    """
    # Each candidate is (repo_path, org_name, provider_value).
    candidates: list[tuple[str, str, str]] = []

    # 1. Selected repository owner first. This entry doubles as the legacy
    #    single ``org_config`` (org_configs[0]) for agent-servers that predate
    #    the list API, so it must keep the pre-list "selected repo's org skills"
    #    semantics regardless of which global repos happen to resolve.
    if selected_repository and len(selected_repository.split('/')) >= 2:
        try:
            org_openhands_repo, org_name = await _determine_org_repo_path(
                selected_repository, user_context
            )
            selected_provider = await _get_provider_type(
                selected_repository, user_context
            )
            candidates.append((org_openhands_repo, org_name, selected_provider))
            if selected_provider == 'github':
                candidates.append((f'{org_name}/.agents', org_name, selected_provider))
        except Exception as e:
            _logger.debug(f'Failed to determine selected-repo org config: {e}')

    # 2. Global owners: the user's account plus their orgs/groups, per provider.
    try:
        provider_handler = await user_context.get_provider_handler()
        for provider in provider_handler.provider_tokens:
            owners = await _enumerate_owners_for_provider(provider_handler, provider)
            for owner in owners:
                for path in _candidate_repo_paths(provider, owner):
                    candidates.append((path, owner, provider.value))
    except Exception as e:
        _logger.debug(f'Failed to enumerate global skill repos: {e}')

    if not candidates:
        return []

    # 3. Deduplicate by repo path (covers owner == login and org overlaps).
    #    Dedup is path-only, so the rare case of a same-named owner on two
    #    providers collapses to the first provider's config; acceptable because
    #    the resolved authenticated URL determines which repo is actually cloned.
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for path, org_name, provider_value in candidates:
        if path not in seen:
            seen.add(path)
            unique.append((path, org_name, provider_value))

    # 4. Bound the verification fan-out. Because the selected-repo entry is first,
    #    truncation here can never drop it.
    if len(unique) > _MAX_ORG_CANDIDATES:
        _logger.warning(
            f'Truncating org skill candidates from {len(unique)} to '
            f'{_MAX_ORG_CANDIDATES}'
        )
        unique = unique[:_MAX_ORG_CANDIDATES]

    # 5. Resolve authenticated URLs concurrently (bounded); keep repos that exist.
    sem = asyncio.Semaphore(_URL_RESOLVE_CONCURRENCY)

    async def _resolve(path: str) -> str | None:
        async with sem:
            return await _get_org_repository_url(path, user_context)

    urls = await asyncio.gather(*[_resolve(path) for path, _, _ in unique])

    configs: list[OrgConfig] = []
    for (path, org_name, provider_value), org_repo_url in zip(
        unique, urls, strict=True
    ):
        if org_repo_url:
            configs.append(
                OrgConfig(
                    repository=path,
                    provider=provider_value,
                    org_repo_url=org_repo_url,
                    org_name=org_name,
                )
            )
    return configs


def parse_marketplace_source(source: str) -> tuple[str, str]:
    """Parse marketplace source into provider and repo path.

    Args:
        source: Marketplace source (e.g., 'github:owner/repo', 'gitlab:owner/repo',
                'https://github.com/owner/repo.git')

    Returns:
        Tuple of (provider, repo_path) where provider is 'github', 'gitlab', etc.
    """
    # Handle github:owner/repo format
    if source.startswith('github:'):
        return ('github', source[7:].lstrip('/'))
    if source.startswith('gitlab:'):
        return ('gitlab', source[7:].lstrip('/'))
    if source.startswith('bitbucket:'):
        return ('bitbucket', source[10:].lstrip('/'))
    if source.startswith('azure-devops:'):
        return ('azure-devops', source[13:].lstrip('/'))

    # Handle URL format
    lower = source.lower()
    if 'github.com' in lower:
        path = source.split('github.com', 1)[1].lstrip('/').rstrip('/')
        # Remove .git suffix if present (use removesuffix to avoid character-by-character stripping)
        if path.endswith('.git'):
            path = path[:-4]
        return ('github', path)
    if 'gitlab.com' in lower or 'gitlab' in lower:
        path = (
            source.split(('gitlab.com' if 'gitlab.com' in lower else 'gitlab'), 1)[1]
            .lstrip('/')
            .rstrip('/')
        )
        if path.endswith('.git'):
            path = path[:-4]
        return ('gitlab', path)
    if 'bitbucket.org' in lower:
        path = source.split('bitbucket.org', 1)[1].lstrip('/').rstrip('/')
        if path.endswith('.git'):
            path = path[:-4]
        return ('bitbucket', path)

    # Default to github
    path = source.rstrip('/')
    if path.endswith('.git'):
        path = path[:-4]
    return ('github', path)


async def authenticate_marketplace_sources(
    registered_marketplaces: list[MarketplaceRegistration] | None,
    user_context: UserContext,
) -> list[MarketplaceRegistration] | None:
    """Swap auto-load marketplace sources for authenticated git URLs.

    The agent-server clones ``auto_load`` marketplace registrations itself but
    has no provider credentials, so private repositories fail to clone (and a
    bare ``owner/repo`` source is misread as a nonexistent local path). Resolve
    each such source to an authenticated URL via the user's provider tokens and
    return copies with ``source`` replaced. Registrations that cannot be
    resolved (local sources, missing tokens, inaccessible repos) pass through
    unchanged and failures never raise, so public/local behavior is unaffected.

    The rewritten URLs embed credentials (like ``OrgConfig.org_repo_url``):
    never log or persist them.

    Args:
        registered_marketplaces: Composed marketplace registrations, or None.
        user_context: UserContext to resolve authenticated URLs.

    Returns:
        New list with rewritten copies (input not mutated), or the input
        itself when None/empty.
    """
    if not registered_marketplaces:
        return registered_marketplaces

    sem = asyncio.Semaphore(_URL_RESOLVE_CONCURRENCY)

    async def _authenticate(reg: MarketplaceRegistration) -> MarketplaceRegistration:
        # Only auto_load registrations are cloned during /api/skills.
        if not reg.auto_load:
            return reg
        _, repo_name = parse_marketplace_source(reg.source)
        # Skip sources that are not owner/repo-shaped: single-segment local
        # paths and URLs on hosts that did not map to a provider repo path.
        if not repo_name or '/' not in repo_name or '://' in repo_name:
            return reg
        try:
            async with sem:
                url = await user_context.get_authenticated_git_url(
                    repo_name, is_optional=True
                )
        except Exception as e:
            _logger.debug(
                f'Could not resolve authenticated URL for marketplace '
                f'{reg.name} ({repo_name}); keeping original source: {e}'
            )
            return reg
        # model_copy() intentionally skips re-validation: validate_source
        # rejects credentialed URLs, and the rewritten value is wire-only
        # (never persisted or returned via the settings API).
        return reg.model_copy(update={'source': url})

    return list(
        await asyncio.gather(*[_authenticate(reg) for reg in registered_marketplaces])
    )


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
    org_configs: list[OrgConfig] | None = None,
    sandbox_config: SandboxConfig | None = None,
    load_public: bool = True,
    load_user: bool = True,
    load_project: bool = True,
    load_org: bool = True,
    registered_marketplaces: list[MarketplaceRegistration] | None = None,
) -> list[Skill]:
    """Load all skills from the agent-server.

    This function makes a single API call to the agent-server's /api/skills
    endpoint to load and merge skills from all configured sources.

    Args:
        agent_server_url: URL of the agent server (e.g., 'http://localhost:8000')
        session_api_key: Session API key for authentication (optional)
        project_dir: Workspace directory path for project skills
        org_configs: Organization/user skill repositories to load (optional)
        sandbox_config: Sandbox skills configuration (optional)
        load_public: Whether to load public skills (default: True)
        load_user: Whether to load user skills (default: True)
        load_project: Whether to load project skills (default: True)
        load_org: Whether to load organization skills (default: True)
        registered_marketplaces: List of marketplace registrations (optional)

    Returns:
        List of Skill objects merged from all sources.
        Returns empty list on error.
    """
    try:
        # Build request payload. ``org_configs`` is the current list form;
        # ``org_config`` (the first entry) is kept for backward compatibility
        # with older agent-server images that only understand a single config.
        payload: dict[str, Any] = {
            'load_public': load_public,
            'load_user': load_user,
            'load_project': load_project,
            'load_org': load_org,
            'project_dir': project_dir,
            'org_configs': (
                [c.model_dump() for c in org_configs] if org_configs else None
            ),
            'org_config': org_configs[0].model_dump() if org_configs else None,
            'sandbox_config': sandbox_config.model_dump() if sandbox_config else None,
        }

        # Only include ``registered_marketplaces`` when we actually have a value.
        # The agent-server field is a non-Optional ``list`` with a default, so an
        # explicit ``null`` fails validation (422) and would drop *all* skills.
        # Omitting the key lets the agent-server apply its own default, and older
        # agent-servers without the field simply ignore an absent key.
        if registered_marketplaces is not None:
            payload['registered_marketplaces'] = [
                reg.model_dump(exclude_none=True, exclude={'scope'})
                for reg in registered_marketplaces
            ]

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
    trigger: TaskTrigger | KeywordTrigger | None = None

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
