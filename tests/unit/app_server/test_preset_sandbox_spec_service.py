"""Tests for PresetSandboxSpecService."""

import pytest

from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo


@pytest.fixture
def service() -> PresetSandboxSpecService:
    return PresetSandboxSpecService(
        specs=[
            SandboxSpecInfo(id='spec-1', command=['/bin/sh']),
            SandboxSpecInfo(id='spec-2', command=['/bin/sh']),
            SandboxSpecInfo(id='spec-3', command=['/bin/sh']),
        ]
    )


@pytest.mark.asyncio
async def test_search_sandbox_specs_invalid_page_id_starts_from_beginning(
    service: PresetSandboxSpecService,
):
    page = await service.search_sandbox_specs(page_id='invalid', limit=2)

    assert [spec.id for spec in page.items] == ['spec-1', 'spec-2']
    assert page.next_page_id == '2'


@pytest.mark.asyncio
async def test_search_sandbox_specs_negative_page_id_starts_from_beginning(
    service: PresetSandboxSpecService,
):
    page = await service.search_sandbox_specs(page_id='-1', limit=2)

    assert [spec.id for spec in page.items] == ['spec-1', 'spec-2']
    assert page.next_page_id == '2'
