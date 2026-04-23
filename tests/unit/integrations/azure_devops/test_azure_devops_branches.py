from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.integrations.azure_devops.azure_devops_service import AzureDevOpsService


@pytest.mark.asyncio
async def test_search_branches_azure_devops_supports_pagination():
    service = AzureDevOpsService(token=SecretStr('t'))

    refs_response = {
        'value': [
            {'name': 'refs/heads/feature/a', 'objectId': 'aaa'},
            {'name': 'refs/heads/feature/b', 'objectId': 'bbb'},
            {'name': 'refs/heads/chore/c', 'objectId': 'ccc'},
        ]
    }

    responses = [
        (refs_response, {}),
        ({'committer': {'date': '2024-01-01T00:00:00Z'}}, {}),
        ({'committer': {'date': '2024-01-02T00:00:00Z'}}, {}),
    ]

    with patch.object(service, '_make_request', side_effect=responses):
        branches = await service.search_branches(
            'org/project/repo', query='feature', per_page=1, page=2
        )

    assert len(branches) == 1
    assert branches[0].name == 'feature/b'
    assert branches[0].commit_sha == 'bbb'
    assert branches[0].last_push_date == '2024-01-02T00:00:00Z'