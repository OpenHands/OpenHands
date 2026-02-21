from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
    PluginSpec,
    redact_url_credentials,
)
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.sdk import Event


@pytest.mark.asyncio
async def test_app_conversation_start_request_polymorphism():
    class MyCallbackProcessor(EventCallbackProcessor):
        async def __call__(
            self,
            conversation_id: UUID,
            callback: EventCallback,
            event: Event,
        ) -> EventCallbackResult | None:
            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail='Live long and prosper!',
            )

    req = AppConversationStartRequest(processors=[MyCallbackProcessor()])
    assert len(req.processors) == 1
    processor = req.processors[0]
    result = await processor(uuid4(), MagicMock(id=uuid4()), MagicMock(id=str(uuid4())))
    assert result.detail == 'Live long and prosper!'


# --- Credential redaction tests (issue #12959) ---


class TestRedactUrlCredentials:
    """Tests for the redact_url_credentials utility function."""

    def test_redacts_oauth_token(self):
        url = 'https://oauth2:ghp_abc123SECRET@github.com/org/repo.git'
        result = redact_url_credentials(url)
        assert 'ghp_abc123SECRET' not in result
        assert 'oauth2:****@github.com' in result

    def test_redacts_user_password(self):
        url = 'https://user:s3cretP@ss@gitlab.com/org/repo'
        result = redact_url_credentials(url)
        assert 's3cretP@ss' not in result
        assert 'user:****@gitlab.com' in result

    def test_preserves_url_without_credentials(self):
        url = 'https://github.com/org/repo.git'
        assert redact_url_credentials(url) == url

    def test_preserves_github_shorthand(self):
        source = 'github:owner/repo'
        assert redact_url_credentials(source) == source

    def test_preserves_local_path(self):
        source = '/home/user/plugins/my-plugin'
        assert redact_url_credentials(source) == source

    def test_preserves_url_with_port(self):
        url = 'https://user:token@gitlab.example.com:8443/org/repo'
        result = redact_url_credentials(url)
        assert 'token' not in result
        assert ':8443' in result
        assert 'user:****@gitlab.example.com:8443' in result

    def test_preserves_username_only(self):
        url = 'https://user@github.com/org/repo'
        result = redact_url_credentials(url)
        assert result == url  # no password to redact

    def test_preserves_path_and_query(self):
        url = 'https://user:secret@host.com/path/to/repo?ref=main'
        result = redact_url_credentials(url)
        assert 'secret' not in result
        assert '/path/to/repo' in result
        assert 'ref=main' in result


class TestPluginSpecRedaction:
    """Tests that PluginSpec redacts credentials during serialization."""

    def test_model_dump_redacts_source(self):
        plugin = PluginSpec(
            source='https://oauth2:SECRET_TOKEN@gitlab.com/org/repo',
            ref='main',
        )
        dumped = plugin.model_dump()
        assert 'SECRET_TOKEN' not in dumped['source']
        assert '****' in dumped['source']

    def test_model_dump_json_redacts_source(self):
        plugin = PluginSpec(
            source='https://oauth2:SECRET_TOKEN@gitlab.com/org/repo',
            ref='main',
        )
        json_str = plugin.model_dump_json()
        assert 'SECRET_TOKEN' not in json_str
        assert '****' in json_str

    def test_in_memory_source_preserved(self):
        """The original URL with credentials must remain accessible in memory
        for runtime plugin fetch operations."""
        plugin = PluginSpec(
            source='https://oauth2:SECRET_TOKEN@gitlab.com/org/repo',
            ref='main',
        )
        assert 'SECRET_TOKEN' in plugin.source

    def test_start_request_redacts_plugin_source(self):
        """End-to-end: credentials are redacted when AppConversationStartRequest
        is serialized (the path that feeds API responses and DB storage)."""
        request = AppConversationStartRequest(
            plugins=[
                PluginSpec(
                    source='https://oauth2:MY_SECRET@gitlab.com/org/repo',
                    ref='v1.0',
                )
            ]
        )
        dumped = request.model_dump()
        plugin_source = dumped['plugins'][0]['source']
        assert 'MY_SECRET' not in plugin_source
        assert '****' in plugin_source

    def test_start_task_redacts_nested_plugin_source(self):
        """Full chain: AppConversationStartTask -> request -> plugins -> source
        is redacted during serialization."""
        task = AppConversationStartTask(
            created_by_user_id='user-1',
            status=AppConversationStartTaskStatus.WORKING,
            request=AppConversationStartRequest(
                plugins=[
                    PluginSpec(
                        source='https://oauth2:TOP_SECRET@gitlab.com/org/repo',
                        ref='main',
                    )
                ]
            ),
        )
        json_str = task.model_dump_json()
        assert 'TOP_SECRET' not in json_str
        assert '****' in json_str
