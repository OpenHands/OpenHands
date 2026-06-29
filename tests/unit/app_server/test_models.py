from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    ACP_SERVER_TAG_KEY,
    AppConversation,
    AppConversationInfo,
    AppConversationStartRequest,
    AppConversationUpdateRequest,
    PluginSpec,
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


def test_app_conversation_update_request_includes_title_field():
    """Test that AppConversationUpdateRequest supports updating the title field.

    The frontend sends a 'title' field when renaming conversations via
    PATCH /api/v1/app-conversations/{id}. The backend model must include
    this field so that title updates are not silently ignored.

    This test verifies that:
    1. The title field exists in the model
    2. When title is provided, it appears in model_fields_set
    3. The title value can be retrieved from the request object

    The service layer uses model_fields_set to determine which fields to update,
    so if title is not in model_fields_set, the update will be silently ignored.
    """
    # Simulate what the frontend sends when renaming a conversation
    update_data = {'title': 'My New Conversation Title'}
    request = AppConversationUpdateRequest.model_validate(update_data)

    # The title field must be recognized and tracked in model_fields_set
    assert 'title' in request.model_fields_set, (
        'title field is not in model_fields_set - title updates will be silently ignored! '
        "Add 'title: str | None = None' to AppConversationUpdateRequest."
    )

    # The title value must be accessible
    assert request.title == 'My New Conversation Title'


def test_app_conversation_update_request_title_field_updates_conversation_info():
    """Test that title from update request can be applied to AppConversationInfo.

    This simulates the service layer logic that iterates over model_fields_set
    and applies each field to the conversation info object.
    """
    # Create a conversation info with default title
    info = AppConversationInfo(
        created_by_user_id='user-123',
        sandbox_id='sandbox-456',
        title='Original Title',
    )

    # Create an update request with a new title
    request = AppConversationUpdateRequest(title='Updated Title')

    # Simulate the service layer update logic
    for field_name in request.model_fields_set:
        value = getattr(request, field_name)
        setattr(info, field_name, value)

    # Verify the title was updated
    assert info.title == 'Updated Title', (
        'Title was not updated on AppConversationInfo. '
        "Ensure 'title' is in AppConversationUpdateRequest.model_fields_set."
    )


class TestAcpServerProjection:
    """``AppConversationInfo.acp_server`` projects the ``acpserver`` tag.

    The conversation UI on the cloud backend reads this top-level field to
    resolve a provider brand label and render the model picker.
    """

    def _acp_info(self, **overrides) -> AppConversationInfo:
        kwargs: dict = {
            'created_by_user_id': 'u',
            'sandbox_id': 'sb',
            'agent_kind': 'acp',
            'tags': {ACP_SERVER_TAG_KEY: 'codex'},
        }
        kwargs.update(overrides)
        return AppConversationInfo(**kwargs)

    def test_projects_provider_key_from_tag(self):
        assert self._acp_info().acp_server == 'codex'

    def test_serialized_into_json_for_the_canvas(self):
        info = self._acp_info()
        assert info.model_dump()['acp_server'] == 'codex'
        assert info.model_dump_json().find('"acp_server":"codex"') != -1

    def test_survives_db_roundtrip_via_tags_only(self):
        # The list endpoint rebuilds info from stored columns; ``tags`` is the
        # only carrier, so re-deriving from it alone must still yield the key.
        info = self._acp_info()
        restored = AppConversationInfo(
            created_by_user_id=None,
            sandbox_id='sb',
            agent_kind='acp',
            tags=info.tags,
        )
        assert restored.acp_server == 'codex'

    def test_survives_app_conversation_build(self):
        # ``_build_conversation`` does ``AppConversation(**info.model_dump())``.
        info = self._acp_info()
        conv = AppConversation(**info.model_dump())
        assert conv.acp_server == 'codex'

    def test_none_for_openhands_even_with_stray_tag(self):
        info = self._acp_info(agent_kind='openhands')
        assert info.acp_server is None

    def test_none_when_tag_absent(self):
        info = self._acp_info(tags={})
        assert info.acp_server is None


class TestPluginSpecSourceRedaction:
    """Verify PluginSpec.source credentials are redacted on serialization."""

    def test_model_dump_redacts_credentials(self):
        spec = PluginSpec(source='https://oauth2:SECRET@gitlab.com/org/repo.git')
        dumped = spec.model_dump()
        assert '****' in dumped['source']
        assert 'SECRET' not in dumped['source']

    def test_model_dump_json_redacts_credentials(self):
        spec = PluginSpec(source='https://user:pass@github.com/org/repo.git')
        json_str = spec.model_dump_json()
        assert 'pass' not in json_str
        assert '****' in json_str

    def test_source_attribute_retains_raw_value(self):
        url = 'https://oauth2:SECRET@gitlab.com/org/repo.git'
        spec = PluginSpec(source=url)
        assert spec.source == url

    def test_clean_url_passes_through_unchanged(self):
        url = 'https://github.com/org/repo.git'
        spec = PluginSpec(source=url)
        assert spec.model_dump()['source'] == url

    def test_redaction_nested_in_start_request(self):
        """Credentials stay out of model_dump when PluginSpec is nested."""
        req = AppConversationStartRequest(
            plugins=[PluginSpec(source='https://token@github.com/org/repo.git')]
        )
        dumped = req.model_dump()
        plugin_source = dumped['plugins'][0]['source']
        assert 'token' not in plugin_source
        assert '****' in plugin_source
