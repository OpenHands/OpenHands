from unittest.mock import MagicMock

from openhands.server.conversation_manager.docker_nested_conversation_manager import (
    DockerNestedConversationManager,
)


def test_get_docker_client_reinitializes_when_none(monkeypatch):
    manager = DockerNestedConversationManager(
        sio=MagicMock(),
        config=MagicMock(),
        server_config=MagicMock(),
        file_store=MagicMock(),
        docker_client=None,
    )
    expected_client = MagicMock()
    monkeypatch.setattr(
        'openhands.server.conversation_manager.docker_nested_conversation_manager.docker.from_env',
        lambda: expected_client,
    )

    client = manager._get_docker_client()

    assert client is expected_client
    assert manager.docker_client is expected_client


def test_get_docker_client_uses_existing_client(monkeypatch):
    existing_client = MagicMock()
    manager = DockerNestedConversationManager(
        sio=MagicMock(),
        config=MagicMock(),
        server_config=MagicMock(),
        file_store=MagicMock(),
        docker_client=existing_client,
    )

    from_env_calls = {'count': 0}

    def _from_env():
        from_env_calls['count'] += 1
        return MagicMock()

    monkeypatch.setattr(
        'openhands.server.conversation_manager.docker_nested_conversation_manager.docker.from_env',
        _from_env,
    )

    client = manager._get_docker_client()

    assert client is existing_client
    assert from_env_calls['count'] == 0
