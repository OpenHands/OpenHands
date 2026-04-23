from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.forgejo.forgejo_service import ForgejoService
from openhands.integrations.service_types import Branch


@pytest.mark.asyncio
async def test_search_branches_forgejo_supports_pagination():
    service = ForgejoService(token=SecretStr('t'))

    branches = [
        Branch(name='feature/a', commit_sha='aaa', protected=False),
        Branch(name='feature/b', commit_sha='bbb', protected=False),
        Branch(name='chore/c', commit_sha='ccc', protected=False),
    ]

    with patch.object(service, 'get_branches', AsyncMock(return_value=branches)):
        result = await service.search_branches(
            'owner/repo', query='feature', per_page=1, page=2
        )

    assert [branch.name for branch in result] == ['feature/b']