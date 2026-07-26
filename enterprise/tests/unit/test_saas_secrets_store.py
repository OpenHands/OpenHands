import hashlib
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from storage.saas_secrets_store import SaasSecretsStore
from storage.stored_custom_secrets import StoredCustomSecrets

from openhands.app_server.integrations.provider import CustomSecret
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey

_MANAGED_NAME = 'CODEX_AUTH_JSON'
_VERSION_USER_ID = UUID('c3333333-3333-3333-3333-333333333333')
_VERSION_ORG_ID = UUID('a1111111-1111-1111-1111-111111111111')
_OTHER_ORG_ID = UUID('b2222222-2222-2222-2222-222222222222')


def _make_jwt_service() -> JwtService:
    key = EncryptionKey(kid='test', key=SecretStr('test_secret'), active=True)
    return JwtService(keys=[key])


@pytest.fixture
def jwt_svc():
    return _make_jwt_service()


@pytest.fixture
def mock_user():
    """Mock user with org_id."""
    user = MagicMock()
    user.current_org_id = UUID('a1111111-1111-1111-1111-111111111111')
    return user


@pytest.fixture
def secrets_store(async_session_maker, jwt_svc):
    # Inject the test session maker into the store module
    import storage.saas_secrets_store as store_module

    store_module.a_session_maker = async_session_maker

    store = SaasSecretsStore('user-id', jwt_svc)
    # Also add it as an attribute for tests that need direct access
    store.a_session_maker = async_session_maker
    return store


@pytest.fixture
def version_store(async_session_maker, jwt_svc):
    import storage.saas_secrets_store as store_module

    store_module.a_session_maker = async_session_maker
    return SaasSecretsStore(str(_VERSION_USER_ID), jwt_svc)


@pytest.fixture
def version_membership():
    with patch(
        'storage.saas_secrets_store.OrgMemberStore.get_org_member',
        new_callable=AsyncMock,
        return_value=MagicMock(),
    ) as get_org_member:
        yield get_org_member


async def _insert_versioned(
    async_session_maker,
    jwt_svc: JwtService,
    value: str,
    org_id: UUID = _VERSION_ORG_ID,
    description: str = 'Managed login',
) -> None:
    async with async_session_maker() as session:
        session.add(
            StoredCustomSecrets(
                keycloak_user_id=str(_VERSION_USER_ID),
                org_id=org_id,
                secret_name=_MANAGED_NAME,
                secret_value=jwt_svc.encrypt_value(value),
                description=jwt_svc.encrypt_value(description),
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_versioned_compare_and_swap(
    async_session_maker,
    jwt_svc,
    version_store,
    version_membership,
):
    original = '{"tokens":{"refresh_token":"r0"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    await _insert_versioned(async_session_maker, jwt_svc, original)

    value, version = await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    )
    assert value == original
    assert version != hashlib.sha256(original.encode()).hexdigest()

    with pytest.raises(CredentialVersionConflict):
        await version_store.replace_versioned(
            _MANAGED_NAME,
            'stale',
            rotated,
            _VERSION_ORG_ID,
        )
    successor = await version_store.replace_versioned(
        _MANAGED_NAME,
        version,
        rotated,
        _VERSION_ORG_ID,
    )

    assert successor != version
    assert await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    ) == (rotated, successor)


@pytest.mark.asyncio
async def test_versioned_newest_duplicate_wins_and_replace_converges_rows(
    async_session_maker,
    jwt_svc,
    version_store,
    version_membership,
):
    await _insert_versioned(
        async_session_maker,
        jwt_svc,
        'stale',
        description='First',
    )
    await _insert_versioned(
        async_session_maker,
        jwt_svc,
        'current',
        description='Newest',
    )

    value, version = await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    )
    assert value == 'current'
    successor = await version_store.replace_versioned(
        _MANAGED_NAME,
        version,
        'rotated',
        _VERSION_ORG_ID,
    )

    async with async_session_maker() as session:
        result = await session.execute(
            select(StoredCustomSecrets)
            .filter(
                StoredCustomSecrets.keycloak_user_id == str(_VERSION_USER_ID),
                StoredCustomSecrets.org_id == _VERSION_ORG_ID,
                StoredCustomSecrets.secret_name == _MANAGED_NAME,
            )
            .order_by(StoredCustomSecrets.id)
        )
        rows = result.scalars().all()

    assert len(rows) == 2
    assert {jwt_svc.decrypt_value(row.secret_value) for row in rows} == {'rotated'}
    assert {jwt_svc.decrypt_value(row.description) for row in rows} == {
        'First',
        'Newest',
    }
    assert await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    ) == ('rotated', successor)


@pytest.mark.asyncio
async def test_versioned_delete_and_identical_recreate_rejects_old_generation(
    async_session_maker,
    jwt_svc,
    version_store,
    version_membership,
):
    await _insert_versioned(async_session_maker, jwt_svc, 'same')
    user = MagicMock(current_org_id=_VERSION_ORG_ID)
    with patch(
        'storage.saas_secrets_store.UserStore.get_user_by_id',
        new_callable=AsyncMock,
        return_value=user,
    ):
        assert await version_store.load() is not None
        _, original_version = await version_store.load_versioned(
            _MANAGED_NAME,
            _VERSION_ORG_ID,
        )
        await version_store.store(Secrets())
        await version_store.store(
            Secrets(
                custom_secrets={
                    _MANAGED_NAME: CustomSecret.from_value(
                        {'secret': 'same', 'description': 'Managed login'}
                    )
                }
            )
        )

    _, recreated_version = await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    )
    assert recreated_version != original_version
    with pytest.raises(CredentialVersionConflict):
        await version_store.replace_versioned(
            _MANAGED_NAME,
            original_version,
            'rotated',
            _VERSION_ORG_ID,
        )


@pytest.mark.asyncio
async def test_versioned_access_uses_pinned_org_after_current_org_drift(
    async_session_maker,
    version_store,
    version_membership,
):
    await _insert_versioned(
        async_session_maker,
        version_store._jwt_svc,
        'value',
    )
    version_store.effective_org_id = _OTHER_ORG_ID

    value, _ = await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    )
    assert value == 'value'
    version_membership.assert_awaited_once_with(
        _VERSION_ORG_ID,
        _VERSION_USER_ID,
    )

    version_membership.reset_mock()
    version_membership.return_value = None
    with pytest.raises(KeyError, match=_MANAGED_NAME):
        await version_store.load_versioned(_MANAGED_NAME, _VERSION_ORG_ID)
    version_membership.assert_awaited_once_with(
        _VERSION_ORG_ID,
        _VERSION_USER_ID,
    )


@pytest.mark.asyncio
@patch(
    'storage.saas_secrets_store.UserStore.get_user_by_id',
    new_callable=AsyncMock,
)
async def test_stale_whole_save_preserves_rotation_and_unrelated_edits(
    mock_get_user,
    async_session_maker,
    jwt_svc,
    version_store,
    version_membership,
):
    user = MagicMock(current_org_id=_VERSION_ORG_ID)
    mock_get_user.return_value = user
    original = '{"tokens":{"refresh_token":"r0"}}'
    rotated = '{"tokens":{"refresh_token":"r1"}}'
    await version_store.store(
        Secrets(
            custom_secrets={
                _MANAGED_NAME: CustomSecret.from_value(
                    {'secret': original, 'description': 'Old description'}
                ),
                'OTHER': CustomSecret.from_value({'secret': 'old', 'description': ''}),
            }
        )
    )
    stale_store = SaasSecretsStore(str(_VERSION_USER_ID), jwt_svc)
    stale = await stale_store.load()
    assert stale is not None
    _, version = await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    )
    successor = await version_store.replace_versioned(
        _MANAGED_NAME,
        version,
        rotated,
        _VERSION_ORG_ID,
    )

    updated = dict(stale.custom_secrets)
    updated[_MANAGED_NAME] = CustomSecret.from_value(
        {'secret': original, 'description': 'New description'}
    )
    updated['OTHER'] = CustomSecret.from_value({'secret': 'new', 'description': ''})
    await stale_store.store(stale.model_copy(update={'custom_secrets': updated}))

    assert await version_store.load_versioned(
        _MANAGED_NAME,
        _VERSION_ORG_ID,
    ) == (rotated, successor)
    loaded = await stale_store.load()
    assert loaded is not None
    assert loaded.custom_secrets[_MANAGED_NAME].description == 'New description'
    assert loaded.custom_secrets['OTHER'].secret.get_secret_value() == 'new'


@pytest.mark.asyncio
async def test_versioned_replace_reports_concurrent_recreate(
    jwt_svc,
    version_membership,
):
    first_result = MagicMock()
    first_result.scalars.return_value.all.return_value = []
    second_result = MagicMock()
    second_result.scalars.return_value.first.return_value = MagicMock()
    session = AsyncMock()
    session.execute.side_effect = [first_result, second_result]
    session_maker = MagicMock()
    session_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch('storage.saas_secrets_store.a_session_maker', session_maker):
        with pytest.raises(CredentialVersionConflict):
            await SaasSecretsStore(
                str(_VERSION_USER_ID),
                jwt_svc,
            ).replace_versioned(
                _MANAGED_NAME,
                'stale',
                'rotated',
                _VERSION_ORG_ID,
            )

    assert session.execute.await_count == 2
    assert session.execute.await_args_list[0].args[0]._for_update_arg is not None


class TestSaasSecretsStore:
    @pytest.mark.asyncio
    @patch(
        'storage.saas_secrets_store.UserStore.get_user_by_id',
        new_callable=AsyncMock,
    )
    async def test_store_and_load(self, mock_get_user, secrets_store, mock_user):
        # Setup mock
        mock_get_user.return_value = mock_user

        # Create a Secrets object with some test data
        user_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'secret_api_token', 'description': ''}
                    ),
                    'db_password': CustomSecret.from_value(
                        {'secret': 'my_password', 'description': ''}
                    ),
                }
            )
        )

        # Store the secrets
        await secrets_store.store(user_secrets)

        # Load the secrets back
        loaded_secrets = await secrets_store.load()

        # Verify the loaded secrets match the original
        assert loaded_secrets is not None
        assert (
            loaded_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'secret_api_token'
        )
        assert (
            loaded_secrets.custom_secrets['db_password'].secret.get_secret_value()
            == 'my_password'
        )

    @pytest.mark.asyncio
    @patch(
        'storage.saas_secrets_store.UserStore.get_user_by_id',
        new_callable=AsyncMock,
    )
    async def test_encryption_decryption(self, mock_get_user, secrets_store, mock_user):
        # Setup mock
        mock_get_user.return_value = mock_user
        # Create a Secrets object with sensitive data
        user_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'sensitive_token', 'description': ''}
                    ),
                    'secret_key': CustomSecret.from_value(
                        {'secret': 'sensitive_secret', 'description': ''}
                    ),
                    'normal_data': CustomSecret.from_value(
                        {'secret': 'not_sensitive', 'description': ''}
                    ),
                }
            )
        )

        assert (
            user_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'sensitive_token'
        )
        # Store the secrets
        await secrets_store.store(user_secrets)

        # Verify the data is encrypted in the database
        from sqlalchemy import select

        async with secrets_store.a_session_maker() as session:
            result = await session.execute(
                select(StoredCustomSecrets)
                .filter(StoredCustomSecrets.keycloak_user_id == 'user-id')
                .filter(StoredCustomSecrets.org_id == mock_user.current_org_id)
            )
            stored = result.scalars().first()

            # The sensitive data should be encrypted
            assert stored.secret_value != 'sensitive_token'
            assert stored.secret_value != 'sensitive_secret'
            assert stored.secret_value != 'not_sensitive'

        # Load the secrets and verify decryption works
        loaded_secrets = await secrets_store.load()
        assert (
            loaded_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'sensitive_token'
        )
        assert (
            loaded_secrets.custom_secrets['secret_key'].secret.get_secret_value()
            == 'sensitive_secret'
        )
        assert (
            loaded_secrets.custom_secrets['normal_data'].secret.get_secret_value()
            == 'not_sensitive'
        )

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_kwargs(self, secrets_store):
        # Test encryption and decryption directly
        test_data: dict[str, Any] = {
            'api_token': 'test_token',
            'client_secret': 'test_secret',
            'normal_data': 'not_sensitive',
            'nested': {
                'nested_token': 'nested_secret_value',
                'nested_normal': 'nested_normal_value',
            },
        }

        # Encrypt the data
        secrets_store._encrypt_kwargs(test_data)

        # Sensitive data is encrypted
        assert test_data['api_token'] != 'test_token'
        assert test_data['client_secret'] != 'test_secret'
        assert test_data['normal_data'] != 'not_sensitive'
        assert test_data['nested']['nested_token'] != 'nested_secret_value'
        assert test_data['nested']['nested_normal'] != 'nested_normal_value'

        # Decrypt the data
        secrets_store._decrypt_kwargs(test_data)

        # Verify sensitive data is properly decrypted
        assert test_data['api_token'] == 'test_token'
        assert test_data['client_secret'] == 'test_secret'
        assert test_data['normal_data'] == 'not_sensitive'
        assert test_data['nested']['nested_token'] == 'nested_secret_value'
        assert test_data['nested']['nested_normal'] == 'nested_normal_value'

    @pytest.mark.asyncio
    async def test_empty_user_id(self, secrets_store):
        # Test that load returns None when user_id is empty
        secrets_store.user_id = ''
        assert await secrets_store.load() is None

    @pytest.mark.asyncio
    @patch(
        'storage.saas_secrets_store.UserStore.get_user_by_id',
        new_callable=AsyncMock,
    )
    async def test_update_existing_secrets(
        self, mock_get_user, secrets_store, mock_user
    ):
        # Setup mock
        mock_get_user.return_value = mock_user
        # Create and store initial secrets
        initial_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'initial_token', 'description': ''}
                    ),
                    'other_value': CustomSecret.from_value(
                        {'secret': 'initial_value', 'description': ''}
                    ),
                }
            )
        )
        await secrets_store.store(initial_secrets)

        # Create and store updated secrets
        updated_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'api_token': CustomSecret.from_value(
                        {'secret': 'updated_token', 'description': ''}
                    ),
                    'new_value': CustomSecret.from_value(
                        {'secret': 'new_value', 'description': ''}
                    ),
                }
            )
        )
        await secrets_store.store(updated_secrets)

        # Load the secrets and verify they were updated
        loaded_secrets = await secrets_store.load()
        assert (
            loaded_secrets.custom_secrets['api_token'].secret.get_secret_value()
            == 'updated_token'
        )
        assert 'new_value' in loaded_secrets.custom_secrets
        assert (
            loaded_secrets.custom_secrets['new_value'].secret.get_secret_value()
            == 'new_value'
        )

        # The other_value should not still be present
        assert 'other_value' not in loaded_secrets.custom_secrets

    @pytest.mark.asyncio
    async def test_get_instance(self, jwt_svc):
        # Test the get_instance class method
        with patch('storage.encrypt_utils.get_jwt_service', return_value=jwt_svc):
            store = await SaasSecretsStore.get_instance('test-user-id')
        assert isinstance(store, SaasSecretsStore)
        assert store.user_id == 'test-user-id'
        assert store._jwt_svc is jwt_svc

    @pytest.mark.asyncio
    @patch(
        'storage.saas_secrets_store.UserStore.get_user_by_id',
        new_callable=AsyncMock,
    )
    async def test_secrets_isolation_between_organizations(
        self, mock_get_user, secrets_store, mock_user
    ):
        org1_id = UUID('a1111111-1111-1111-1111-111111111111')
        org2_id = UUID('b2222222-2222-2222-2222-222222222222')

        # Store secrets in org1 (personal workspace)
        mock_user.current_org_id = org1_id
        mock_get_user.return_value = mock_user
        org1_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'personal_secret': CustomSecret.from_value(
                        {
                            'secret': 'personal_secret_value',
                            'description': 'My personal secret',
                        }
                    ),
                }
            )
        )
        await secrets_store.store(org1_secrets)

        # Verify org1 secrets are stored
        loaded_org1 = await secrets_store.load()
        assert loaded_org1 is not None
        assert 'personal_secret' in loaded_org1.custom_secrets
        assert (
            loaded_org1.custom_secrets['personal_secret'].secret.get_secret_value()
            == 'personal_secret_value'
        )

        # Switch to org2 and store secrets there
        mock_user.current_org_id = org2_id
        mock_get_user.return_value = mock_user
        org2_secrets = Secrets(
            custom_secrets=MappingProxyType(
                {
                    'org2_secret': CustomSecret.from_value(
                        {'secret': 'org2_secret_value', 'description': 'Org2 secret'}
                    ),
                }
            )
        )
        await secrets_store.store(org2_secrets)

        # Verify org2 secrets are stored
        loaded_org2 = await secrets_store.load()
        assert loaded_org2 is not None
        assert 'org2_secret' in loaded_org2.custom_secrets
        assert (
            loaded_org2.custom_secrets['org2_secret'].secret.get_secret_value()
            == 'org2_secret_value'
        )

        # Switch back to org1 and verify secrets are still there
        mock_user.current_org_id = org1_id
        mock_get_user.return_value = mock_user
        loaded_org1_again = await secrets_store.load()
        assert loaded_org1_again is not None
        assert 'personal_secret' in loaded_org1_again.custom_secrets
        assert (
            loaded_org1_again.custom_secrets[
                'personal_secret'
            ].secret.get_secret_value()
            == 'personal_secret_value'
        )
        # Verify org2 secrets are NOT visible in org1
        assert 'org2_secret' not in loaded_org1_again.custom_secrets
