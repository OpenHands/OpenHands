"""Tests for PresetSandboxSpecService – Round 7 additions.

Covers:
- empty-specs guard in get_default_sandbox_spec
- negative page_id clamped to 0 in search_sandbox_specs
"""

import pytest

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo


def _make_spec(id_: str) -> SandboxSpecInfo:
    return SandboxSpecInfo(id=id_, command=None)


# ---------------------------------------------------------------------------
# get_default_sandbox_spec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_default_sandbox_spec_returns_first_when_specs_present():
    spec_a = _make_spec('ubuntu')
    spec_b = _make_spec('alpine')
    service = PresetSandboxSpecService(specs=[spec_a, spec_b])
    result = await service.get_default_sandbox_spec()
    assert result is spec_a


@pytest.mark.asyncio
async def test_get_default_sandbox_spec_raises_sandbox_error_when_empty():
    service = PresetSandboxSpecService(specs=[])
    with pytest.raises(SandboxError, match='No sandbox specs configured'):
        await service.get_default_sandbox_spec()


# ---------------------------------------------------------------------------
# search_sandbox_specs – negative page_id clamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_sandbox_specs_negative_page_id_clamped_to_zero():
    specs = [_make_spec(f'img{i}') for i in range(5)]
    service = PresetSandboxSpecService(specs=specs)
    page = await service.search_sandbox_specs(page_id='-3', limit=100)
    # Negative index must be treated as 0, returning all items from the start
    assert page.items == specs


@pytest.mark.asyncio
async def test_search_sandbox_specs_zero_page_id_returns_from_start():
    specs = [_make_spec(f'img{i}') for i in range(3)]
    service = PresetSandboxSpecService(specs=specs)
    page = await service.search_sandbox_specs(page_id='0', limit=100)
    assert page.items == specs
