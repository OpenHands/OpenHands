import json
from unittest.mock import MagicMock

import pytest

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.integrations.provider import ProviderType
from openhands.app_server.secrets.file_secrets_store import FileSecretsStore


@pytest.mark.asyncio
async def test_load_filters_null_provider_token_entries():
    file_store = MagicMock(spec=FileStore)
    file_store.read.return_value = json.dumps(
        {
            'provider_tokens': {
                'github': None,
                'gitlab': {'token': None},
                'bitbucket': {'token': 'bitbucket-token'},
            },
            'custom_secrets': {},
        }
    )
    store = FileSecretsStore(file_store)

    secrets = await store.load()

    assert secrets is not None
    assert ProviderType.GITHUB not in secrets.provider_tokens
    assert ProviderType.GITLAB not in secrets.provider_tokens
    assert ProviderType.BITBUCKET in secrets.provider_tokens
