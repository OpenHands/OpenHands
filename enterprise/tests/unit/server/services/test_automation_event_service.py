"""
Unit tests for AutomationEventService.

Tests the service that forwards GitHub webhook events to the automation service.

The service is optimized for high-traffic with:
- Redis caching for org claim lookups (1 hour TTL)
- Redis caching for provider→Keycloak user ID mappings (24 hour TTL)
- Lazy access control (membership checks deferred to execution time)
- Separate AUTOMATION_WEBHOOK_SECRET for internal service communication
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.integrations.service_types import ProviderType

# Default patches for constants
CONSTANT_PATCHES = {
    'server.services.automation_event_service.AUTOMATION_WEBHOOK_SECRET': 'test-shared-secret',
    'server.services.automation_event_service.AUTOMATION_SERVICE_TIMEOUT': 30,
}


@pytest.fixture
def mock_token_manager():
    """Create a mock TokenManager."""
    return MagicMock()


@pytest.fixture
def mock_org_git_claim():
    """Create a mock OrgGitClaim."""
    claim = MagicMock()
    claim.org_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return claim


@pytest.fixture
def github_org_payload():
    """Create a sample GitHub webhook payload for an organization repo."""
    return {
        'repository': {
            'id': 123456,
            'full_name': 'test-org/test-repo',
            'private': False,
            'default_branch': 'main',
            'owner': {
                'login': 'test-org',
                'id': 789,
                'type': 'Organization',
            },
        },
        'sender': {
            'id': 12345,
            'login': 'testuser',
        },
        'action': 'opened',
        'installation': {
            'id': 99999,
        },
    }


@pytest.fixture
def github_user_payload():
    """Create a sample GitHub webhook payload for a personal/user repo."""
    return {
        'repository': {
            'id': 654321,
            'full_name': 'testuser/personal-repo',
            'private': True,
            'default_branch': 'main',
            'owner': {
                'login': 'testuser',
                'id': 12345,
                'type': 'User',
            },
        },
        'sender': {
            'id': 12345,
            'login': 'testuser',
        },
        'action': 'opened',
        'installation': {
            'id': 99999,
        },
    }


def create_service(mock_token_manager):
    """Helper to create a service with mocked sio and constants."""
    with patch('server.services.automation_event_service.sio'), patch.dict(
        'os.environ', {}, clear=False
    ):
        for key, value in CONSTANT_PATCHES.items():
            patch(key, value).start()

        from server.services.automation_event_service import AutomationEventService

        return AutomationEventService(mock_token_manager)


class TestResolveGitOrg:
    """Tests for _resolve_git_org method with caching."""

    @pytest.mark.asyncio
    async def test_resolve_git_org_cache_miss_found(
        self, mock_token_manager, mock_org_git_claim
    ):
        """
        GIVEN: Cache miss and org claim exists in DB
        WHEN: _resolve_git_org is called
        THEN: Org ID is returned and cached
        """
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)  # Cache miss
        mock_redis.setex = AsyncMock()

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
            return_value=mock_org_git_claim.org_id,
        ), patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._resolve_git_org(ProviderType.GITHUB, 'test-org')

            assert result == mock_org_git_claim.org_id
            # Verify result was cached
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_git_org_cache_hit(self, mock_token_manager):
        """
        GIVEN: Org ID is cached in Redis
        WHEN: _resolve_git_org is called
        THEN: Cached value is returned without calling resolve_org_for_repo
        """
        cached_org_id = '12345678-1234-5678-1234-567812345678'
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=cached_org_id.encode())

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
        ) as mock_resolver, patch(
            'server.services.automation_event_service.sio'
        ) as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._resolve_git_org(ProviderType.GITHUB, 'test-org')

            assert result == uuid.UUID(cached_org_id)
            # resolve_org_for_repo should NOT be called
            mock_resolver.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_git_org_cache_miss_not_found(self, mock_token_manager):
        """
        GIVEN: Cache miss and org claim does NOT exist in DB
        WHEN: _resolve_git_org is called
        THEN: None is returned and negative result is cached
        """
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)  # Cache miss
        mock_redis.setex = AsyncMock()

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
            return_value=None,
        ), patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._resolve_git_org(
                ProviderType.GITHUB, 'unclaimed-org'
            )

            assert result is None
            # Verify negative result was cached
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            # Second positional arg is the value
            assert call_args[0][2] == 'none'  # Negative cache value

    @pytest.mark.asyncio
    async def test_resolve_git_org_negative_cache_hit(self, mock_token_manager):
        """
        GIVEN: Negative result is cached (org not claimed)
        WHEN: _resolve_git_org is called
        THEN: None is returned without calling resolve_org_for_repo
        """
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b'none')  # Cached negative

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
        ) as mock_resolver, patch(
            'server.services.automation_event_service.sio'
        ) as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._resolve_git_org(
                ProviderType.GITHUB, 'unclaimed-org'
            )

            assert result is None
            mock_resolver.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_git_org_includes_provider_in_cache_key(
        self, mock_token_manager, mock_org_git_claim
    ):
        """
        GIVEN: GitHub provider with an org name
        WHEN: _resolve_git_org is called
        THEN: Cache key includes the provider name
        """
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
            return_value=mock_org_git_claim.org_id,
        ), patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)

            # Call for GitHub
            await service._resolve_git_org(ProviderType.GITHUB, 'test-org')
            github_cache_key = mock_redis.setex.call_args_list[0][0][0]

            # Cache key should include provider
            assert 'github' in github_cache_key


class TestResolvePersonalOrg:
    """Tests for _resolve_personal_org method with caching."""

    @pytest.mark.asyncio
    async def test_resolve_personal_org_cache_miss_found(self, mock_token_manager):
        """
        GIVEN: Cache miss and user exists in Keycloak
        WHEN: _resolve_personal_org is called
        THEN: Keycloak ID is returned and cached
        """
        keycloak_id = '87654321-4321-8765-4321-876543218765'
        mock_token_manager.get_user_id_from_idp_user_id = AsyncMock(
            return_value=keycloak_id
        )

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)  # Cache miss
        mock_redis.setex = AsyncMock()

        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._resolve_personal_org(ProviderType.GITHUB, 12345)

            assert result == uuid.UUID(keycloak_id)
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_personal_org_cache_hit(self, mock_token_manager):
        """
        GIVEN: Keycloak ID is cached in Redis
        WHEN: _resolve_personal_org is called
        THEN: Cached value is returned without Keycloak query
        """
        keycloak_id = '87654321-4321-8765-4321-876543218765'
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=keycloak_id.encode())

        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._resolve_personal_org(ProviderType.GITHUB, 12345)

            assert result == uuid.UUID(keycloak_id)
            # Token manager should NOT be called
            mock_token_manager.get_user_id_from_idp_user_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_personal_org_no_user_id(self, mock_token_manager):
        """
        GIVEN: No provider user ID provided
        WHEN: _resolve_personal_org is called
        THEN: None is returned immediately
        """
        service = create_service(mock_token_manager)
        result = await service._resolve_personal_org(ProviderType.GITHUB, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_personal_org_includes_provider_in_cache_key(
        self, mock_token_manager
    ):
        """
        GIVEN: GitHub provider with a user ID
        WHEN: _resolve_personal_org is called
        THEN: Cache key includes the provider name
        """
        keycloak_id = '87654321-4321-8765-4321-876543218765'
        mock_token_manager.get_user_id_from_idp_user_id = AsyncMock(
            return_value=keycloak_id
        )

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)

            # Call for GitHub
            await service._resolve_personal_org(ProviderType.GITHUB, 12345)
            github_cache_key = mock_redis.setex.call_args_list[0][0][0]

            # Cache key should include provider
            assert 'github' in github_cache_key


class TestForwardEvent:
    """Tests for forward_event method (minimal payload, no access control)."""

    @pytest.mark.asyncio
    async def test_forward_org_event_success(
        self, mock_token_manager, github_org_payload, mock_org_git_claim
    ):
        """
        GIVEN: A GitHub event from a claimed organization repo
        WHEN: forward_event is called
        THEN: Minimal payload is forwarded (no access_control)
        """
        from server.services.automation_event_service import AutomationEventService

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
            return_value=mock_org_git_claim.org_id,
        ), patch(
            'server.services.automation_event_service.sio'
        ) as mock_sio, patch.object(
            AutomationEventService,
            '_send_to_automation_service',
            new_callable=AsyncMock,
        ) as mock_send:
            mock_sio.manager.redis = mock_redis

            service = AutomationEventService(mock_token_manager)
            await service.forward_event(
                provider=ProviderType.GITHUB,
                payload=github_org_payload,
                installation_id=99999,
            )

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            # Provider is first arg, org_id is second
            assert call_args[0][0] == ProviderType.GITHUB
            assert call_args[0][1] == mock_org_git_claim.org_id

            payload = call_args[0][2]
            assert payload['organization']['git_org'] == 'test-org'
            assert 'payload' in payload
            # access_control should NOT be in payload (lazy evaluation)
            assert 'access_control' not in payload

    @pytest.mark.asyncio
    async def test_forward_personal_repo_event_success(
        self, mock_token_manager, github_user_payload
    ):
        """
        GIVEN: A GitHub event from a personal repo with linked OpenHands account
        WHEN: forward_event is called
        THEN: Event is forwarded using the user's personal org (keycloak ID)
        """
        from server.services.automation_event_service import AutomationEventService

        keycloak_id = '87654321-4321-8765-4321-876543218765'
        mock_token_manager.get_user_id_from_idp_user_id = AsyncMock(
            return_value=keycloak_id
        )

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
            return_value=None,  # No org claim for personal repo
        ), patch(
            'server.services.automation_event_service.sio'
        ) as mock_sio, patch.object(
            AutomationEventService,
            '_send_to_automation_service',
            new_callable=AsyncMock,
        ) as mock_send:
            mock_sio.manager.redis = mock_redis

            service = AutomationEventService(mock_token_manager)
            await service.forward_event(
                provider=ProviderType.GITHUB,
                payload=github_user_payload,
                installation_id=99999,
            )

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            # Provider is first arg, org_id is second (personal org = keycloak ID)
            assert call_args[0][0] == ProviderType.GITHUB
            assert call_args[0][1] == uuid.UUID(keycloak_id)
            payload = call_args[0][2]
            assert payload['organization']['git_org'] == 'testuser'
            assert payload['organization']['openhands_org_id'] == keycloak_id

    @pytest.mark.asyncio
    async def test_forward_event_no_owner_in_payload(self, mock_token_manager):
        """
        GIVEN: A GitHub event with no repository owner in payload
        WHEN: forward_event is called
        THEN: Event is skipped with warning log
        """
        from server.services.automation_event_service import AutomationEventService

        payload = {
            'repository': {},
            'sender': {'id': 12345, 'login': 'testuser'},
        }

        with patch('server.services.automation_event_service.sio'), patch(
            'server.services.automation_event_service.logger'
        ) as mock_logger, patch.object(
            AutomationEventService,
            '_send_to_automation_service',
            new_callable=AsyncMock,
        ) as mock_send:
            service = AutomationEventService(mock_token_manager)
            await service.forward_event(
                provider=ProviderType.GITHUB,
                payload=payload,
                installation_id=99999,
            )

            mock_send.assert_not_called()
            mock_logger.warning.assert_called()
            assert 'No repository owner' in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_forward_event_org_not_claimed_and_not_personal(
        self, mock_token_manager, github_org_payload
    ):
        """
        GIVEN: A GitHub event from an org that isn't claimed (and isn't personal)
        WHEN: forward_event is called
        THEN: Event is skipped with warning log
        """
        from server.services.automation_event_service import AutomationEventService

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch(
            'server.services.automation_event_service.resolve_org_for_repo',
            new_callable=AsyncMock,
            return_value=None,
        ), patch('server.services.automation_event_service.sio') as mock_sio, patch(
            'server.services.automation_event_service.logger'
        ) as mock_logger, patch.object(
            AutomationEventService,
            '_send_to_automation_service',
            new_callable=AsyncMock,
        ) as mock_send:
            mock_sio.manager.redis = mock_redis

            service = AutomationEventService(mock_token_manager)
            await service.forward_event(
                provider=ProviderType.GITHUB,
                payload=github_org_payload,
                installation_id=99999,
            )

            mock_send.assert_not_called()
            mock_logger.warning.assert_called()
            assert 'not claimed' in str(mock_logger.warning.call_args)


class TestExtractOwnerInfo:
    """Tests for _extract_owner_info method."""

    def test_extract_github_owner_info(self, mock_token_manager, github_org_payload):
        """
        GIVEN: A GitHub webhook payload
        WHEN: _extract_owner_info is called
        THEN: GitHub owner info is correctly extracted
        """
        service = create_service(mock_token_manager)
        git_org, owner_type, owner_id = service._extract_owner_info(
            ProviderType.GITHUB, github_org_payload
        )

        assert git_org == 'test-org'
        assert owner_type == 'Organization'
        assert owner_id == 789

    def test_extract_github_user_owner_info(
        self, mock_token_manager, github_user_payload
    ):
        """
        GIVEN: A GitHub webhook payload from a personal repo
        WHEN: _extract_owner_info is called
        THEN: User owner info is correctly extracted
        """
        service = create_service(mock_token_manager)
        git_org, owner_type, owner_id = service._extract_owner_info(
            ProviderType.GITHUB, github_user_payload
        )

        assert git_org == 'testuser'
        assert owner_type == 'User'
        assert owner_id == 12345


class TestBuildEventPayload:
    """Tests for _build_event_payload method."""

    def test_build_minimal_payload(self, mock_token_manager):
        """
        GIVEN: Org context and payload
        WHEN: _build_event_payload is called
        THEN: Minimal payload with only org + payload is returned
        """
        from server.services.automation_event_service import OrgContext

        service = create_service(mock_token_manager)

        org_context = OrgContext(
            org_id=uuid.UUID('12345678-1234-5678-1234-567812345678'),
            git_org='test-org',
        )
        test_payload = {'action': 'opened', 'sender': {'login': 'user'}}

        result = service._build_event_payload(org_context, test_payload)

        assert result == {
            'organization': {
                'git_org': 'test-org',
                'openhands_org_id': '12345678-1234-5678-1234-567812345678',
            },
            'payload': test_payload,
        }
        # Verify NO access_control in payload
        assert 'access_control' not in result


class TestSendToAutomationService:
    """Tests for _send_to_automation_service method."""

    @pytest.mark.asyncio
    async def test_send_success(self, mock_token_manager):
        """
        GIVEN: AUTOMATION_SERVICE_URL is configured
        WHEN: _send_to_automation_service is called
        THEN: Request is sent with correct signature and provider in URL
        """

        org_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
        payload = {'organization': {'git_org': 'test'}, 'payload': {}}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'matched': 2})

        mock_post_context = MagicMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_context.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=mock_post_context)

        mock_session_context = MagicMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            'server.services.automation_event_service.AUTOMATION_SERVICE_URL',
            'https://automation.example.com',
        ), patch('server.services.automation_event_service.sio'), patch(
            'server.services.automation_event_service.aiohttp.ClientSession',
            return_value=mock_session_context,
        ):
            service = create_service(mock_token_manager)
            await service._send_to_automation_service(
                ProviderType.GITHUB, org_id, payload
            )

            # Verify the POST was called
            mock_session_instance.post.assert_called_once()
            # Verify URL includes provider
            call_args = mock_session_instance.post.call_args
            url = call_args[0][0]
            assert '/github' in url
            assert str(org_id) in url

    @pytest.mark.asyncio
    async def test_send_includes_provider_in_url(self, mock_token_manager):
        """
        GIVEN: GitHub provider
        WHEN: _send_to_automation_service is called
        THEN: The URL includes the provider name
        """
        org_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
        payload = {}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'matched': 0})

        mock_post_context = MagicMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_context.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=mock_post_context)

        mock_session_context = MagicMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            'server.services.automation_event_service.AUTOMATION_SERVICE_URL',
            'https://automation.example.com',
        ), patch('server.services.automation_event_service.sio'), patch(
            'server.services.automation_event_service.aiohttp.ClientSession',
            return_value=mock_session_context,
        ):
            service = create_service(mock_token_manager)

            # Test GitHub
            await service._send_to_automation_service(
                ProviderType.GITHUB, org_id, payload
            )
            github_url = mock_session_instance.post.call_args_list[0][0][0]
            assert github_url.endswith('/github')

    @pytest.mark.asyncio
    async def test_send_no_url_configured(self, mock_token_manager):
        """
        GIVEN: AUTOMATION_SERVICE_URL is not configured
        WHEN: _send_to_automation_service is called
        THEN: Warning is logged and nothing is sent
        """
        org_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
        payload = {}

        with patch(
            'server.services.automation_event_service.AUTOMATION_SERVICE_URL', None
        ), patch('server.services.automation_event_service.sio'), patch(
            'server.services.automation_event_service.logger'
        ) as mock_logger:
            service = create_service(mock_token_manager)
            await service._send_to_automation_service(
                ProviderType.GITHUB, org_id, payload
            )

            mock_logger.warning.assert_called()
            assert 'not configured' in str(mock_logger.warning.call_args)


class TestSignPayload:
    """Tests for _sign_payload method."""

    def test_sign_payload(self, mock_token_manager):
        """
        GIVEN: A payload bytes
        WHEN: _sign_payload is called
        THEN: HMAC-SHA256 signature is returned in correct format
        """
        with patch(
            'server.services.automation_event_service.AUTOMATION_WEBHOOK_SECRET',
            'test-shared-secret',
        ), patch('server.services.automation_event_service.sio'):
            service = create_service(mock_token_manager)
            payload_bytes = b'{"test": "data"}'

            signature = service._sign_payload(payload_bytes)

            assert signature.startswith('sha256=')
            assert len(signature) == 71  # 'sha256=' + 64 hex chars

    def test_sign_payload_uses_dedicated_secret(self, mock_token_manager):
        """
        GIVEN: AUTOMATION_WEBHOOK_SECRET is configured
        WHEN: _sign_payload is called
        THEN: The dedicated secret is used (not GitHub webhook secret)
        """
        import hashlib
        import hmac

        # Use the default test secret from CONSTANT_PATCHES
        shared_secret = 'test-shared-secret'
        payload_bytes = b'{"test": "data"}'

        # Calculate expected signature with the shared secret
        expected_sig = hmac.new(
            shared_secret.encode('utf-8'),
            msg=payload_bytes,
            digestmod=hashlib.sha256,
        ).hexdigest()

        with patch(
            'server.services.automation_event_service.AUTOMATION_WEBHOOK_SECRET',
            shared_secret,
        ), patch('server.services.automation_event_service.sio'):
            service = create_service(mock_token_manager)
            signature = service._sign_payload(payload_bytes)

            assert signature == f'sha256={expected_sig}'


class TestCacheHelpers:
    """Tests for generic cache helper methods."""

    @pytest.mark.asyncio
    async def test_get_cached_value_hit(self, mock_token_manager):
        """
        GIVEN: Value exists in Redis cache
        WHEN: _get_cached_value is called
        THEN: Decoded string value is returned
        """
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b'cached-value')

        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._get_cached_value('test-key')

            assert result == 'cached-value'

    @pytest.mark.asyncio
    async def test_get_cached_value_miss(self, mock_token_manager):
        """
        GIVEN: Value does not exist in Redis cache
        WHEN: _get_cached_value is called
        THEN: None is returned
        """
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            result = await service._get_cached_value('test-key')

            assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_value_redis_unavailable(self, mock_token_manager):
        """
        GIVEN: Redis is unavailable
        WHEN: _get_cached_value is called
        THEN: None is returned (graceful degradation)
        """
        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = None

            service = create_service(mock_token_manager)
            result = await service._get_cached_value('test-key')

            assert result is None

    @pytest.mark.asyncio
    async def test_set_cached_value_success(self, mock_token_manager):
        """
        GIVEN: Redis is available
        WHEN: _set_cached_value is called
        THEN: Value is stored with TTL
        """
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()

        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = mock_redis

            service = create_service(mock_token_manager)
            await service._set_cached_value('test-key', 'test-value', 3600)

            mock_redis.setex.assert_called_once_with('test-key', 3600, 'test-value')

    @pytest.mark.asyncio
    async def test_set_cached_value_redis_unavailable(self, mock_token_manager):
        """
        GIVEN: Redis is unavailable
        WHEN: _set_cached_value is called
        THEN: No error is raised (silent failure)
        """
        with patch('server.services.automation_event_service.sio') as mock_sio:
            mock_sio.manager.redis = None

            service = create_service(mock_token_manager)
            # Should not raise
            await service._set_cached_value('test-key', 'test-value', 3600)
