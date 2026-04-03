from pathlib import Path

import pytest

from openhands.app_server.user import skills_router


def _write_skill(directory: Path, name: str, *, frontmatter_name: str | None = None):
    skill_name = frontmatter_name or name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.md').write_text(
        f"---\nname: {skill_name}\ntype: knowledge\ntriggers:\n  - demo\n---\n# {skill_name}\n",
        encoding='utf-8',
    )


@pytest.mark.asyncio
class TestSearchSkillsPagination:
    async def test_next_page_id_is_unique_across_sources(self, tmp_path, monkeypatch):
        global_dir = tmp_path / 'global'
        user_dir = tmp_path / 'user'
        _write_skill(global_dir, 'alpha')
        _write_skill(global_dir, 'shared')
        _write_skill(user_dir, 'shared')

        monkeypatch.setattr(skills_router, 'GLOBAL_SKILLS_DIR', global_dir)
        monkeypatch.setattr(skills_router, 'USER_SKILLS_DIR', user_dir)

        first_page = await skills_router.search_skills(limit=2)

        assert [skill.name for skill in first_page.items] == ['alpha', 'shared']
        assert first_page.next_page_id == 'global:shared'

        second_page = await skills_router.search_skills(
            page_id=first_page.next_page_id, limit=2
        )

        assert [skill.name for skill in second_page.items] == ['shared']
        assert [skill.source for skill in second_page.items] == ['user']
        assert second_page.next_page_id is None

    async def test_legacy_name_only_page_id_still_advances(self, tmp_path, monkeypatch):
        global_dir = tmp_path / 'global'
        user_dir = tmp_path / 'user'
        _write_skill(global_dir, 'alpha')
        _write_skill(global_dir, 'shared')
        _write_skill(user_dir, 'beta')

        monkeypatch.setattr(skills_router, 'GLOBAL_SKILLS_DIR', global_dir)
        monkeypatch.setattr(skills_router, 'USER_SKILLS_DIR', user_dir)

        page = await skills_router.search_skills(page_id='shared', limit=2)

        assert [skill.name for skill in page.items] == ['beta']
        assert [skill.source for skill in page.items] == ['user']
