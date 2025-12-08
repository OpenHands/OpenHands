from types import MappingProxyType

import pytest
from pydantic import SecretStr, ValidationError

from openhands.events.action.commands import CmdRunAction
from openhands.integrations.provider import (
    ProviderHandler,
    ProviderToken,
    ProviderType,
)
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings


def test_provider_token_immutability():
    """Test that ProviderToken is immutable"""
    token = ProviderToken(token=SecretStr('test'), user_id='user1')

    # Test direct attribute modification
    with pytest.raises(ValidationError):
        token.token = SecretStr('new')

    with pytest.raises(ValidationError):
        token.user_id = 'new_user'

    # Test that __setattr__ is blocked
    with pytest.raises(ValidationError):
        setattr(token, 'token', SecretStr('new'))

    # Verify original values are unchanged
    assert token.token.get_secret_value() == 'test'
    assert token.user_id == 'user1'


def test_secret_store_immutability():
    """Test that Secrets is immutable"""
    store = Secrets(
        provider_tokens={ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))}
    )

    # Test direct attribute modification
    with pytest.raises(ValidationError):
        store.provider_tokens = {}

    # Test dictionary mutation attempts
    with pytest.raises((TypeError, AttributeError)):
        store.provider_tokens[ProviderType.GITHUB] = ProviderToken(
            token=SecretStr('new')
        )

    with pytest.raises((TypeError, AttributeError)):
        store.provider_tokens.clear()

    with pytest.raises((TypeError, AttributeError)):
        store.provider_tokens.update(
            {ProviderType.GITLAB: ProviderToken(token=SecretStr('test'))}
        )

    # Test nested immutability
    github_token = store.provider_tokens[ProviderType.GITHUB]
    with pytest.raises(ValidationError):
        github_token.token = SecretStr('new')

    # Verify original values are unchanged
    assert store.provider_tokens[ProviderType.GITHUB].token.get_secret_value() == 'test'


def test_settings_immutability():
    """Test that Settings secrets_store is immutable"""
    settings = Settings(
        secrets_store=Secrets(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))
            }
        )
    )

    # Test direct modification of secrets_store
    with pytest.raises(ValidationError):
        settings.secrets_store = Secrets()

    # Test nested modification attempts
    with pytest.raises((TypeError, AttributeError)):
        settings.secrets_store.provider_tokens[ProviderType.GITHUB] = ProviderToken(
            token=SecretStr('new')
        )

    # Test model_copy creates new instance
    new_store = Secrets(
        provider_tokens={
            ProviderType.GITHUB: ProviderToken(token=SecretStr('new_token'))
        }
    )
    new_settings = settings.model_copy(update={'secrets_store': new_store})

    # Verify original is unchanged and new has updated values
    assert (
        settings.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token.get_secret_value()
        == 'test'
    )
    assert (
        new_settings.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token.get_secret_value()
        == 'new_token'
    )

    with pytest.raises(ValidationError):
        new_settings.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token = SecretStr('')


def test_provider_handler_immutability():
    """Test that ProviderHandler maintains token immutability"""
    # Create initial tokens
    tokens = MappingProxyType(
        {ProviderType.GITHUB: ProviderToken(token=SecretStr('test'))}
    )

    handler = ProviderHandler(provider_tokens=tokens)

    # Try to modify tokens (should raise TypeError due to frozen dict)
    with pytest.raises((TypeError, AttributeError)):
        handler.provider_tokens[ProviderType.GITHUB] = ProviderToken(
            token=SecretStr('new')
        )

    # Try to modify the handler's tokens property
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        handler.provider_tokens = {}

    # Original token should be unchanged
    assert (
        handler.provider_tokens[ProviderType.GITHUB].token.get_secret_value() == 'test'
    )


def test_token_conversion():
    """Test token conversion in Secrets.create"""
    # Test with string token
    store1 = Settings(
        secrets_store=Secrets(
            provider_tokens={
                ProviderType.GITHUB: ProviderToken(token=SecretStr('test_token'))
            }
        )
    )

    assert (
        store1.secrets_store.provider_tokens[
            ProviderType.GITHUB
        ].token.get_secret_value()
        == 'test_token'
    )
    assert store1.secrets_store.provider_tokens[ProviderType.GITHUB].user_id is None

    # Test with dict token
    store2 = Secrets(
        provider_tokens={'github': {'token': 'test_token', 'user_id': 'user1'}}
    )
    assert (
        store2.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'test_token'
    )
    assert store2.provider_tokens[ProviderType.GITHUB].user_id == 'user1'

    # Test with ProviderToken
    token = ProviderToken(token=SecretStr('test_token'), user_id='user2')
    store3 = Secrets(provider_tokens={ProviderType.GITHUB: token})
    assert (
        store3.provider_tokens[ProviderType.GITHUB].token.get_secret_value()
        == 'test_token'
    )
    assert store3.provider_tokens[ProviderType.GITHUB].user_id == 'user2'

    store4 = Secrets(
        provider_tokens={
            ProviderType.GITHUB: 123  # Invalid type
        }
    )

    assert ProviderType.GITHUB not in store4.provider_tokens

    # Test with empty/None token
    store5 = Secrets(provider_tokens={ProviderType.GITHUB: None})
    assert ProviderType.GITHUB not in store5.provider_tokens

    store6 = Secrets(
        provider_tokens={
            'invalid_provider': 'test_token'  # Invalid provider type
        }
    )

    assert len(store6.provider_tokens.keys()) == 0


def test_provider_handler_type_enforcement():
    with pytest.raises((TypeError)):
        ProviderHandler(provider_tokens={'a': 'b'})


def test_expose_env_vars():
    """Test that expose_env_vars correctly exposes secrets as strings"""
    tokens = MappingProxyType(
        {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('test_token')),
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
        }
    )
    handler = ProviderHandler(provider_tokens=tokens)

    # Test with specific provider tokens
    env_secrets = {
        ProviderType.GITHUB: SecretStr('gh_token'),
        ProviderType.GITLAB: SecretStr('gl_token'),
    }
    exposed = handler.expose_env_vars(env_secrets)

    assert exposed['github_token'] == 'gh_token'
    assert exposed['gitlab_token'] == 'gl_token'


@pytest.mark.asyncio
async def test_get_env_vars():
    """Test get_env_vars with different configurations"""
    tokens = MappingProxyType(
        {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('test_token')),
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
        }
    )
    handler = ProviderHandler(provider_tokens=tokens)

    # Test getting all tokens unexposed
    env_vars = await handler.get_env_vars(expose_secrets=False)
    assert isinstance(env_vars, dict)
    assert isinstance(env_vars[ProviderType.GITHUB], SecretStr)
    assert env_vars[ProviderType.GITHUB].get_secret_value() == 'test_token'
    assert env_vars[ProviderType.GITLAB].get_secret_value() == 'gitlab_token'

    # Test getting specific providers
    env_vars = await handler.get_env_vars(
        expose_secrets=False, providers=[ProviderType.GITHUB]
    )
    assert len(env_vars) == 1
    assert ProviderType.GITHUB in env_vars
    assert ProviderType.GITLAB not in env_vars

    # Test exposed secrets
    exposed_vars = await handler.get_env_vars(expose_secrets=True)
    assert isinstance(exposed_vars, dict)
    assert exposed_vars['github_token'] == 'test_token'
    assert exposed_vars['gitlab_token'] == 'gitlab_token'

    # Test empty tokens
    empty_handler = ProviderHandler(provider_tokens=MappingProxyType({}))
    empty_vars = await empty_handler.get_env_vars()
    assert empty_vars == {}


@pytest.fixture
def event_stream():
    """Fixture for event stream testing"""

    class TestEventStream:
        def __init__(self):
            self.secrets = {}

        def set_secrets(self, secrets):
            self.secrets = secrets

    return TestEventStream()


@pytest.mark.asyncio
async def test_set_event_stream_secrets(event_stream):
    """Test setting secrets in event stream"""
    tokens = MappingProxyType(
        {
            ProviderType.GITHUB: ProviderToken(token=SecretStr('test_token')),
            ProviderType.GITLAB: ProviderToken(token=SecretStr('gitlab_token')),
        }
    )
    handler = ProviderHandler(provider_tokens=tokens)

    # Test with provided env_vars
    env_vars = {
        ProviderType.GITHUB: SecretStr('new_token'),
        ProviderType.GITLAB: SecretStr('new_gitlab_token'),
    }
    await handler.set_event_stream_secrets(event_stream, env_vars)
    assert event_stream.secrets == {
        'github_token': 'new_token',
        'gitlab_token': 'new_gitlab_token',
    }

    # Test without env_vars (using existing tokens)
    await handler.set_event_stream_secrets(event_stream)
    assert event_stream.secrets == {
        'github_token': 'test_token',
        'gitlab_token': 'gitlab_token',
    }


def test_check_cmd_action_for_provider_token_ref():
    """Test detection of provider tokens in command actions"""
    # Test command with GitHub token
    cmd = CmdRunAction(command='echo $GITHUB_TOKEN')
    providers = ProviderHandler.check_cmd_action_for_provider_token_ref(cmd)
    assert ProviderType.GITHUB in providers
    assert len(providers) == 1

    # Test command with multiple tokens
    cmd = CmdRunAction(command='echo $GITHUB_TOKEN && echo $GITLAB_TOKEN')
    providers = ProviderHandler.check_cmd_action_for_provider_token_ref(cmd)
    assert ProviderType.GITHUB in providers
    assert ProviderType.GITLAB in providers
    assert len(providers) == 2

    # Test command without tokens
    cmd = CmdRunAction(command='echo "Hello"')
    providers = ProviderHandler.check_cmd_action_for_provider_token_ref(cmd)
    assert len(providers) == 0

    # Test non-command action
    from openhands.events.action import MessageAction

    msg = MessageAction(content='test')
    providers = ProviderHandler.check_cmd_action_for_provider_token_ref(msg)
    assert len(providers) == 0


def test_get_provider_env_key():
    """Test provider environment key generation"""
    assert ProviderHandler.get_provider_env_key(ProviderType.GITHUB) == 'github_token'
    assert ProviderHandler.get_provider_env_key(ProviderType.GITLAB) == 'gitlab_token'
