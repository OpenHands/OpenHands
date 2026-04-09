from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Query
from pydantic import BaseModel

from openhands.app_server.utils.dependencies import get_dependencies
from openhands.core.logger import openhands_logger as logger
from openhands.memory.memory import GLOBAL_MICROAGENTS_DIR, USER_MICROAGENTS_DIR

router = APIRouter(prefix='/skills', tags=['Skills'], dependencies=get_dependencies())

# Re-use V0 path constants (single source of truth)
GLOBAL_SKILLS_DIR = Path(GLOBAL_MICROAGENTS_DIR)
USER_SKILLS_DIR = Path(USER_MICROAGENTS_DIR)


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
