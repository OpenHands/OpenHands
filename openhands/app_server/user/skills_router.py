from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, cast

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import openhands
from openhands.app_server.app_conversation.skill_loader import (
    _match_url_source_to_provider,
)
from openhands.app_server.app_conversation.skill_loader import (
    parse_marketplace_source as _parse_marketplace_source,
)
from openhands.app_server.config import depends_user_context
from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
)
from openhands.app_server.settings.settings_models import MarketplaceRegistration
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.sdk.marketplace import Marketplace

router = APIRouter(prefix='/skills', tags=['Skills'], dependencies=get_dependencies())
user_context_dependency = depends_user_context()

# skills/ is at the repo root, two levels above the openhands package __file__
GLOBAL_SKILLS_DIR = Path(openhands.__file__).parent.parent / 'skills'
USER_SKILLS_DIR = Path.home() / '.openhands' / 'microagents'


class SkillInfo(BaseModel):
    """Information about a single available skill."""

    name: str
    type: str  # 'knowledge', 'repo', or 'task'
    source: str  # 'global' or 'user'
    triggers: list[str] | None = None


class SkillPage(BaseModel):
    """Paginated response for the skills search endpoint."""

    items: list[SkillInfo]
    next_page_id: str | None = None


class MarketplacePluginPreview(BaseModel):
    """A plugin advertised by a marketplace manifest.

    The UI operates at the plugin level, so a plugin's bundled skills are not
    expanded here; only the plugin itself is surfaced.
    """

    name: str
    description: str | None = None
    source: str  # the marketplace registration source (e.g. 'github:owner/repo')
    marketplace: str  # the marketplace registration name this plugin belongs to


class MarketplaceSkillsPreviewResponse(BaseModel):
    """Response for marketplace skills preview endpoint."""

    skills: list[SkillInfo]
    plugins: list[MarketplacePluginPreview]
    marketplace_skills: dict[str, list[str]]  # marketplace_name -> skill names
    errors: list[str]


def _parse_skill_frontmatter(file_path: Path) -> dict | None:
    """Parse YAML frontmatter from a skill markdown file.

    Returns the frontmatter dict, or None if parsing fails.
    """
    try:
        text = file_path.read_text(encoding='utf-8')
    except Exception:
        return None

    if not text.startswith('---'):
        return None

    end = text.find('---', 3)
    if end == -1:
        return None

    try:
        return yaml.safe_load(text[3:end])
    except yaml.YAMLError as e:
        logger.warning(f'Invalid YAML frontmatter in {file_path}: {e}')
        return None


def _load_skills_from_dir(skills_dir: Path, source: str) -> list[SkillInfo]:
    """Load skill metadata from a directory of markdown files.

    Args:
        skills_dir: Path to the skills directory.
        source: Source label ('global' or 'user').

    Returns:
        List of SkillInfo objects parsed from the directory.
    """
    skills: list[SkillInfo] = []
    if not skills_dir.exists():
        return skills

    for md_file in skills_dir.rglob('*.md'):
        if md_file.name == 'README.md':
            continue

        try:
            fm = _parse_skill_frontmatter(md_file)
            if not isinstance(fm, dict):
                continue

            # Use name from frontmatter, falling back to filename stem
            name = fm.get('name') or md_file.stem

            # Determine type from frontmatter
            skill_type = fm.get('type', 'knowledge')
            triggers = fm.get('triggers') or None

            skills.append(
                SkillInfo(
                    name=name,
                    type=skill_type,
                    source=source,
                    triggers=triggers,
                )
            )
        except Exception as e:
            logger.warning(f'Failed to parse skill file {md_file}: {e}')

    return skills


@router.get(
    '/search',
    response_model=SkillPage,
)
async def search_skills(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(
            title='The max number of results in the page',
            gt=0,
            le=100,
        ),
    ] = 100,
) -> SkillPage:
    """Search / list available global and user-level skills.

    Returns skill metadata so the frontend can render a toggle list.
    """
    skills: list[SkillInfo] = []

    # Load global skills
    try:
        skills.extend(_load_skills_from_dir(GLOBAL_SKILLS_DIR, 'global'))
    except Exception as e:
        logger.warning(f'Failed to load global skills: {e}')

    # Load user-level skills
    try:
        skills.extend(_load_skills_from_dir(USER_SKILLS_DIR, 'user'))
    except Exception as e:
        logger.warning(f'Failed to load user skills: {e}')

    # Sort by source (global first), then by name
    skills.sort(key=lambda s: (s.source, s.name))

    # Apply cursor-based pagination
    start = 0
    if page_id is not None:
        for i, skill in enumerate(skills):
            if skill.name == page_id:
                start = i + 1
                break

    page = skills[start : start + limit]
    next_page_id = (
        page[-1].name if len(page) == limit and start + limit < len(skills) else None
    )

    return SkillPage(items=page, next_page_id=next_page_id)


async def _clone_marketplace_repo(
    marketplace: MarketplaceRegistration,
    user_context: UserContext,
) -> tuple[Path | None, str]:
    """Clone a marketplace repository to a temporary directory.

    Args:
        marketplace: MarketplaceRegistration with source, ref, and repo_path
        user_context: UserContext for accessing provider tokens

    Returns:
        Tuple of (cloned_path or None, error_message or '')
    """
    provider, repo_path = _parse_marketplace_source(marketplace.source)

    # Validate repo path format
    if not repo_path or '/' not in repo_path:
        return None, f'Invalid repository path: {repo_path}'

    # Authenticate URL/scp sources against the provider whose host matches
    # (pinned), so private self-hosted repos (e.g. Bitbucket DC) clone with
    # credentials. Non-URL owner/repo sources fall through to the public path.
    authenticated_url = None
    try:
        matched = await _match_url_source_to_provider(marketplace.source, user_context)
        if matched is not None:
            matched_provider, matched_repo = matched
            handler = await user_context.get_provider_handler()
            authenticated_url = await handler.get_authenticated_git_url(
                matched_repo, specified_provider=matched_provider
            )
        elif '://' not in repo_path:
            # Bare owner/repo (or recognized public domain): resolve normally.
            provider_tokens = await user_context.get_provider_tokens()
            if provider_tokens:
                typed_provider_tokens = cast(PROVIDER_TOKEN_TYPE, provider_tokens)
                client = ProviderHandler(
                    provider_tokens=MappingProxyType(typed_provider_tokens),
                    external_auth_id=await user_context.get_user_id(),
                )
                authenticated_url = await client.get_authenticated_git_url(repo_path)
    except Exception as e:
        logger.warning(
            f'Failed to get authenticated URL for {repo_path}: {e}, '
            'will try unauthenticated clone'
        )

    if authenticated_url:
        clone_url = authenticated_url
    elif '://' in repo_path:
        # repo_path still carries a scheme, so the source is a full URL on a host
        # _parse_marketplace_source did not recognize AND no configured provider
        # matched it. Refuse rather than git-cloning an arbitrary (possibly
        # attacker-controlled) host (SSRF). Recognized public domains
        # (github.com/gitlab.com/bitbucket.org) reduce to owner/repo above and
        # fall through to the public clone below regardless of login provider.
        return None, f'Unsupported marketplace host: {marketplace.source}'
    else:
        # Public bare owner/repo, recognized provider prefix, or a recognized
        # public-domain URL: clone from the public domain, no auth needed.
        provider_domain_map = {
            'github': 'github.com',
            'gitlab': 'gitlab.com',
            'bitbucket': 'bitbucket.org',
        }
        clone_url = (
            f'https://{provider_domain_map.get(provider, "github.com")}/{repo_path}.git'
        )

    # Create unique temporary directory for this clone using tempfile.mkdtemp
    try:
        clone_dir = Path(
            tempfile.mkdtemp(prefix=f'openhands_marketplace_{marketplace.name}_')
        )

        # Run git without a shell (argv form) and use ``--`` so a source/ref
        # that begins with '-' can never be parsed as a git option (argument
        # injection). Reject leading-'-' values outright as defense in depth.
        if clone_url.startswith('-'):
            _cleanup_clone_dir(clone_dir)
            return None, f'Invalid clone URL: {clone_url}'

        result = subprocess.run(
            ['git', 'clone', '--', clone_url, str(clone_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            _cleanup_clone_dir(clone_dir)
            return None, f'Git clone failed: {result.stderr}'

        # Checkout ref if specified
        if marketplace.ref:
            if marketplace.ref.startswith('-'):
                _cleanup_clone_dir(clone_dir)
                return None, f'Invalid ref: {marketplace.ref}'
            checkout_result = subprocess.run(
                ['git', '-C', str(clone_dir), 'checkout', marketplace.ref],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if checkout_result.returncode != 0:
                _cleanup_clone_dir(clone_dir)
                return None, f'Git checkout failed: {checkout_result.stderr}'

        # Navigate to repo_path if specified
        if marketplace.repo_path:
            skills_path = clone_dir / marketplace.repo_path
            if not skills_path.exists():
                _cleanup_clone_dir(clone_dir)
                return None, f'Repo path not found: {marketplace.repo_path}'
            return skills_path, ''

        return clone_dir, ''

    except subprocess.TimeoutExpired:
        return None, 'Git clone timed out'
    except Exception as e:
        return None, f'Clone failed: {str(e)}'


def _cleanup_clone_dir(clone_dir: Path) -> None:
    """Clean up a cloned repository directory."""
    try:
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
    except Exception as e:
        logger.debug(f'Failed to clean up clone directory {clone_dir}: {e}')


@router.post(
    '/marketplace-skills',
    response_model=MarketplaceSkillsPreviewResponse,
)
async def get_marketplace_skills(
    marketplaces: list[MarketplaceRegistration],
    user_context: UserContext = user_context_dependency,
) -> MarketplaceSkillsPreviewResponse:
    """Get skills from marketplace repositories.

    This endpoint fetches and returns skill metadata from marketplace repos
    without requiring an active sandbox session. Useful for previewing what
    skills a marketplace provides before or after adding it.

    Each call clones the marketplace repos fresh; this endpoint is admin-only
    and low-traffic, so caching the git clones here would buy little for the
    cross-user / pod-local complexity a shared cache introduces.

    Args:
        marketplaces: List of marketplace registrations to fetch skills from.

    Returns:
        MarketplaceSkillsPreviewResponse with skill metadata and any errors.
    """
    all_skills: list[SkillInfo] = []
    plugins: list[MarketplacePluginPreview] = []
    marketplace_skills: dict[str, list[str]] = {}
    errors: list[str] = []

    # Track cloned directories for cleanup
    cloned_dirs: list[Path] = []

    try:
        for marketplace in marketplaces:
            # Clone the marketplace repo
            clone_path, error = await _clone_marketplace_repo(marketplace, user_context)

            if error:
                errors.append(f'{marketplace.name}: {error}')
                continue

            if clone_path is None:
                errors.append(f'{marketplace.name}: Failed to clone repository')
                continue

            cloned_dirs.append(clone_path)

            # Prefer the marketplace manifest so we operate at the *plugin* level.
            # ``Marketplace.load`` parses ``.plugin/marketplace.json`` (or
            # ``.claude-plugin/marketplace.json``) and exposes the plugins and any
            # standalone skills the marketplace advertises. A plugin's bundled
            # skills are intentionally not expanded — the UI shows plugins, not
            # their internals.
            skill_names: list[str] = []
            loaded_marketplace: Marketplace | None = None
            try:
                loaded_marketplace = Marketplace.load(clone_path)
            except FileNotFoundError:
                # No manifest: this is a plain skills repo, not a plugin
                # marketplace. Fall back to a loose-skill scan below.
                loaded_marketplace = None
            except Exception as e:
                logger.warning(
                    f'Failed to parse marketplace manifest for {marketplace.name}: {e}'
                )
                errors.append(f'{marketplace.name}: invalid marketplace manifest')
                loaded_marketplace = None

            if loaded_marketplace is not None:
                for plugin_entry in loaded_marketplace.plugins:
                    plugins.append(
                        MarketplacePluginPreview(
                            name=plugin_entry.name,
                            description=plugin_entry.description,
                            source=marketplace.source,
                            marketplace=marketplace.name,
                        )
                    )
                # Standalone skills declared in the manifest (not plugin-bundled).
                for skill_entry in loaded_marketplace.skills:
                    all_skills.append(
                        SkillInfo(
                            name=skill_entry.name,
                            type='knowledge',
                            source=f'marketplace:{marketplace.name}',
                            triggers=None,
                        )
                    )
                    skill_names.append(skill_entry.name)
            else:
                # No manifest: surface loose skills from skills/ and .skills/.
                # Bundled plugin skills under plugins/*/skills/ are deliberately
                # not flattened — a plugin marketplace should ship a manifest.
                skills_dirs = [
                    d
                    for d in (clone_path / 'skills', clone_path / '.skills')
                    if d.is_dir()
                ]
                for skills_dir in skills_dirs:
                    try:
                        for skill in _load_skills_from_dir(
                            skills_dir, marketplace.source
                        ):
                            all_skills.append(
                                SkillInfo(
                                    name=skill.name,
                                    type=skill.type,
                                    source=f'marketplace:{marketplace.name}',
                                    triggers=skill.triggers,
                                )
                            )
                            skill_names.append(skill.name)
                    except Exception as e:
                        logger.warning(f'Failed to load skills from {skills_dir}: {e}')

            marketplace_skills[marketplace.name] = skill_names

    except Exception as e:
        logger.exception(
            'Unexpected error in marketplace-skills endpoint', stack_info=True
        )
        errors.append(f'Internal error: {str(e)}')
        # Clean up before raising
        for clone_dir in cloned_dirs:
            _cleanup_clone_dir(clone_dir)
        # Raise HTTP 500 for critical errors
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Clean up cloned directories
        for clone_dir in cloned_dirs:
            _cleanup_clone_dir(clone_dir)

    result = MarketplaceSkillsPreviewResponse(
        skills=all_skills,
        plugins=plugins,
        marketplace_skills=marketplace_skills,
        errors=errors,
    )

    return result
