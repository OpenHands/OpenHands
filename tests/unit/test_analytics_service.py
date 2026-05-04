"""Tests for the AnalyticsService and related utilities."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from openhands.analytics import (
    AnalyticsService,
    get_analytics_service,
    init_analytics_service,
)
from openhands.analytics.analytics_constants import (
    CONVERSATION_CREATED,
    CONVERSATION_DELETED,
    CONVERSATION_ERRORED,
    CONVERSATION_FINISHED,
    CREDIT_LIMIT_REACHED,
    CREDIT_PURCHASED,
    GIT_PROVIDER_CONNECTED,
    ONBOARDING_COMPLETED,
    SETTINGS_SAVED,
    TEAM_MEMBERS_INVITED,
    TRAJECTORY_DOWNLOADED,
    USER_LOGGED_IN,
    USER_SIGNED_UP,
)
from openhands.analytics.analytics_context import AnalyticsContext
from openhands.analytics.analytics_service import AnalyticsService as DirectService
from openhands.analytics.oss_install_id import get_or_create_install_id
from openhands.server.types import AppMode


def make_ctx(
    user_id: str = 'user-1',
    consented: bool = True,
    org_id: str | None = None,
) -> AnalyticsContext:
    """Helper to create an AnalyticsContext for tests."""
    return AnalyticsContext(
        user_id=user_id,
        consented=consented,
        org_id=org_id,
        user=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    import openhands.analytics as analytics_module

    original = analytics_module._analytics_service
    analytics_module._analytics_service = None
    yield
    analytics_module._analytics_service = original


@pytest.fixture
def mock_posthog():
    """Patch posthog.Posthog and return the mock instance."""
    with patch(
        'openhands.analytics.analytics_service.Posthog', autospec=True
    ) as MockClass:
        mock_client = MagicMock()
        MockClass.return_value = mock_client
        yield MockClass, mock_client


@pytest.fixture
def oss_service(mock_posthog):
    """AnalyticsService in OSS mode."""
    MockClass, mock_client = mock_posthog
    service = DirectService(
        api_key='test-key',
        host='https://posthog.example.com',
        app_mode=AppMode.OPENHANDS,
        is_feature_env=False,
    )
    return service, mock_client


@pytest.fixture
def saas_service(mock_posthog):
    """AnalyticsService in SaaS mode."""
    MockClass, mock_client = mock_posthog
    service = DirectService(
        api_key='test-key',
        host='https://posthog.example.com',
        app_mode=AppMode.SAAS,
        is_feature_env=False,
    )
    return service, mock_client


@pytest.fixture
def feature_env_service(mock_posthog):
    """AnalyticsService in feature-env mode (OSS)."""
    MockClass, mock_client = mock_posthog
    service = DirectService(
        api_key='test-key',
        host='https://posthog.example.com',
        app_mode=AppMode.OPENHANDS,
        is_feature_env=True,
    )
    return service, mock_client


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_analytics_service_returns_none_before_init(self):
        """get_analytics_service() returns None before init_analytics_service() is called."""
        assert get_analytics_service() is None

    def test_init_analytics_service_returns_instance(self, mock_posthog):
        """init_analytics_service() returns an AnalyticsService instance."""
        service = init_analytics_service(
            api_key='key',
            host='https://posthog.example.com',
            app_mode=AppMode.SAAS,
            is_feature_env=False,
        )
        assert isinstance(service, AnalyticsService)

    def test_get_analytics_service_returns_instance_after_init(self, mock_posthog):
        """get_analytics_service() returns the initialized instance."""
        init_analytics_service(
            api_key='key',
            host='https://posthog.example.com',
            app_mode=AppMode.SAAS,
            is_feature_env=False,
        )
        result = get_analytics_service()
        assert isinstance(result, AnalyticsService)

    def test_init_analytics_service_stores_singleton(self, mock_posthog):
        """Calling init twice returns the same object from get_analytics_service."""
        svc1 = init_analytics_service(
            api_key='key',
            host='https://posthog.example.com',
            app_mode=AppMode.SAAS,
            is_feature_env=False,
        )
        svc2 = get_analytics_service()
        assert svc1 is svc2


# ---------------------------------------------------------------------------
# Consent gate tests
# ---------------------------------------------------------------------------


class TestConsentGate:
    def test_capture_with_consented_false_makes_zero_posthog_calls(self, oss_service):
        """capture() with consented=False produces zero PostHog client calls."""
        service, mock_client = oss_service
        ctx = make_ctx(user_id='user123', consented=False)
        service.capture(
            ctx=ctx,
            event='test event',
        )
        mock_client.capture.assert_not_called()

    def test_capture_with_consented_true_makes_posthog_call(self, oss_service):
        """capture() with consented=True calls the PostHog client."""
        service, mock_client = oss_service
        ctx = make_ctx(user_id='user123', consented=True)
        service.capture(
            ctx=ctx,
            event='test event',
        )
        mock_client.capture.assert_called_once()

    def test_set_person_properties_with_consented_false_is_noop(self, saas_service):
        """set_person_properties() with consented=False does not call SDK."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123', consented=False)
        service.set_person_properties(
            ctx=ctx,
            properties={'name': 'Alice'},
        )
        mock_client.set.assert_not_called()

    def test_group_identify_with_consented_false_is_noop(self, saas_service):
        """group_identify() with consented=False does not call SDK."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123', consented=False)
        service.group_identify(
            ctx=ctx,
            group_type='org',
            group_key='org123',
            properties={'name': 'Acme'},
        )
        mock_client.group_identify.assert_not_called()


# ---------------------------------------------------------------------------
# OSS vs SaaS mode tests
# ---------------------------------------------------------------------------


class TestOssSaasMode:
    def test_capture_oss_mode_includes_process_person_profile_false(self, oss_service):
        """In OSS mode, captured events include '$process_person_profile': False."""
        service, mock_client = oss_service
        ctx = make_ctx(user_id='user123')
        service.capture(
            ctx=ctx,
            event='test event',
        )
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert props.get('$process_person_profile') is False

    def test_capture_saas_mode_does_not_include_process_person_profile(
        self, saas_service
    ):
        """In SaaS mode, captured events do NOT include '$process_person_profile'."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(
            ctx=ctx,
            event='test event',
        )
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert '$process_person_profile' not in props

    def test_set_person_properties_oss_mode_is_noop(self, oss_service):
        """set_person_properties() in OSS mode does not call SDK."""
        service, mock_client = oss_service
        ctx = make_ctx(user_id='user123')
        service.set_person_properties(
            ctx=ctx,
            properties={'name': 'Alice'},
        )
        mock_client.set.assert_not_called()

    def test_set_person_properties_saas_mode_calls_sdk(self, saas_service):
        """set_person_properties() in SaaS mode calls SDK when consented."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.set_person_properties(
            ctx=ctx,
            properties={'name': 'Alice'},
        )
        mock_client.set.assert_called_once()

    def test_group_identify_oss_mode_is_noop(self, oss_service):
        """group_identify() in OSS mode does not call SDK."""
        service, mock_client = oss_service
        ctx = make_ctx(user_id='user123')
        service.group_identify(
            ctx=ctx,
            group_type='org',
            group_key='org123',
            properties={'name': 'Acme'},
        )
        mock_client.group_identify.assert_not_called()

    def test_group_identify_saas_mode_calls_sdk(self, saas_service):
        """group_identify() in SaaS mode calls SDK when consented."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.group_identify(
            ctx=ctx,
            group_type='org',
            group_key='org123',
            properties={'name': 'Acme'},
        )
        mock_client.group_identify.assert_called_once()


# ---------------------------------------------------------------------------
# Common properties tests
# ---------------------------------------------------------------------------


class TestCommonProperties:
    def test_capture_always_includes_app_mode(self, saas_service):
        """Every captured event includes app_mode in properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(ctx=ctx, event='test event')
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert 'app_mode' in props
        assert props['app_mode'] == AppMode.SAAS.value

    def test_capture_always_includes_is_feature_env(self, saas_service):
        """Every captured event includes is_feature_env in properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(ctx=ctx, event='test event')
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert 'is_feature_env' in props
        assert props['is_feature_env'] is False

    def test_capture_includes_org_id_when_provided(self, saas_service):
        """capture() includes org_id when provided."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123', org_id='org-abc')
        service.capture(
            ctx=ctx,
            event='test event',
        )
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert props.get('org_id') == 'org-abc'

    def test_capture_omits_org_id_when_not_provided(self, saas_service):
        """capture() omits org_id when not provided."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(ctx=ctx, event='test event')
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert 'org_id' not in props

    def test_capture_includes_session_id_when_provided(self, saas_service):
        """capture() includes $session_id when session_id provided."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(
            ctx=ctx,
            event='test event',
            session_id='sess-xyz',
        )
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert props.get('$session_id') == 'sess-xyz'

    def test_capture_omits_session_id_when_not_provided(self, saas_service):
        """capture() omits $session_id when not provided."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(ctx=ctx, event='test event')
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert '$session_id' not in props

    def test_capture_merges_caller_properties(self, saas_service):
        """capture() merges caller-provided properties with common ones."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user123')
        service.capture(
            ctx=ctx,
            event='test event',
            properties={'custom_prop': 'custom_val'},
        )
        _, kwargs = mock_client.capture.call_args
        props = kwargs.get('properties', {})
        assert props.get('custom_prop') == 'custom_val'
        assert 'app_mode' in props  # common props still present


# ---------------------------------------------------------------------------
# distinct_id / feature-env tests
# ---------------------------------------------------------------------------


class TestDistinctId:
    def test_distinct_id_normal_mode_returns_raw_user_id(self, saas_service):
        """_distinct_id() returns raw user_id when is_feature_env=False."""
        service, _ = saas_service
        assert service._distinct_id('user123') == 'user123'

    def test_distinct_id_feature_env_returns_prefixed_user_id(
        self, feature_env_service
    ):
        """_distinct_id() returns 'FEATURE_{user_id}' when is_feature_env=True."""
        service, _ = feature_env_service
        assert service._distinct_id('user123') == 'FEATURE_user123'

    def test_capture_uses_distinct_id_helper(self, feature_env_service):
        """capture() passes the prefixed distinct_id to the PostHog client."""
        service, mock_client = feature_env_service
        ctx = make_ctx(user_id='user123')
        service.capture(
            ctx=ctx,
            event='test event',
        )
        _, kwargs = mock_client.capture.call_args
        assert kwargs.get('distinct_id') == 'FEATURE_user123'


# ---------------------------------------------------------------------------
# Shutdown tests
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_calls_client_shutdown(self, saas_service):
        """shutdown() calls client.shutdown() without raising."""
        service, mock_client = saas_service
        service.shutdown()
        mock_client.shutdown.assert_called_once()

    def test_shutdown_logs_errors_without_raising(self, saas_service):
        """shutdown() catches SDK errors and does not propagate them."""
        service, mock_client = saas_service
        mock_client.shutdown.side_effect = RuntimeError('SDK error')
        # Should not raise
        service.shutdown()


# ---------------------------------------------------------------------------
# SDK error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_capture_logs_sdk_errors_without_raising(self, saas_service):
        """capture() catches SDK errors and does not raise to caller."""
        service, mock_client = saas_service
        mock_client.capture.side_effect = RuntimeError('Network error')
        ctx = make_ctx(user_id='user123')
        # Should not raise
        service.capture(ctx=ctx, event='test event')


# ---------------------------------------------------------------------------
# Event constants tests
# ---------------------------------------------------------------------------


class TestEventConstants:
    def test_event_constants_are_lowercase_strings_with_spaces(self):
        """Event constants follow PostHog naming convention (lowercase, spaces)."""
        for const in [
            USER_LOGGED_IN,
            USER_SIGNED_UP,
            CONVERSATION_CREATED,
            CONVERSATION_FINISHED,
            CONVERSATION_ERRORED,
            CREDIT_PURCHASED,
            CREDIT_LIMIT_REACHED,
            GIT_PROVIDER_CONNECTED,
            ONBOARDING_COMPLETED,
        ]:
            assert isinstance(const, str)
            assert const == const.lower(), f'{const!r} is not lowercase'
            assert ' ' in const, f'{const!r} does not contain spaces'

    def test_user_logged_in_constant_value(self):
        """USER_LOGGED_IN has the correct value."""
        assert USER_LOGGED_IN == 'user logged in'

    def test_user_signed_up_constant_value(self):
        """USER_SIGNED_UP has the correct value."""
        assert USER_SIGNED_UP == 'user signed up'


# ---------------------------------------------------------------------------
# OSS install ID tests
# ---------------------------------------------------------------------------


class TestOssInstallId:
    def test_get_or_create_install_id_creates_file_on_first_call(self, tmp_path):
        """get_or_create_install_id() creates the analytics_id.txt file on first call."""
        result = get_or_create_install_id(tmp_path)
        assert (tmp_path / 'analytics_id.txt').exists()
        assert result is not None

    def test_get_or_create_install_id_returns_valid_uuid(self, tmp_path):
        """get_or_create_install_id() returns a valid UUID string."""
        result = get_or_create_install_id(tmp_path)
        parsed = uuid.UUID(result)  # raises if invalid
        assert str(parsed) == result

    def test_get_or_create_install_id_returns_same_uuid_on_second_call(self, tmp_path):
        """get_or_create_install_id() returns the same UUID on subsequent calls."""
        first = get_or_create_install_id(tmp_path)
        second = get_or_create_install_id(tmp_path)
        assert first == second

    def test_get_or_create_install_id_returns_uuid_on_file_write_failure(
        self, tmp_path
    ):
        """get_or_create_install_id() returns an ephemeral UUID when file write fails."""
        read_only_dir = tmp_path / 'readonly'
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # read-only

        try:
            result = get_or_create_install_id(read_only_dir)
            # Should still return a valid UUID, not crash
            parsed = uuid.UUID(result)
            assert str(parsed) == result
        finally:
            read_only_dir.chmod(0o755)  # restore for cleanup


# ---------------------------------------------------------------------------
# identify_user tests
# ---------------------------------------------------------------------------


class TestIdentifyUser:
    """Tests for the identify_user method on AnalyticsService."""

    def test_identify_user_consent_false_is_noop(self, saas_service):
        """identify_user with consented=False is a complete no-op."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1', consented=False, org_id='org-1')
        service.identify_user(
            ctx=ctx,
            email='a@b.com',
        )
        mock_client.set.assert_not_called()
        mock_client.group_identify.assert_not_called()

    def test_identify_user_oss_mode_is_noop(self, oss_service):
        """identify_user in OSS mode is a complete no-op."""
        service, mock_client = oss_service
        ctx = make_ctx(user_id='user-1', org_id='org-1')
        service.identify_user(
            ctx=ctx,
            email='a@b.com',
        )
        mock_client.set.assert_not_called()
        mock_client.group_identify.assert_not_called()

    def test_identify_user_saas_sets_person_properties(self, saas_service):
        """identify_user in SaaS mode with consent calls set_person_properties with expected fields."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1', org_id='org-42')
        service.identify_user(
            ctx=ctx,
            email='alice@example.com',
            org_name='Acme Corp',
            idp='github',
        )
        mock_client.set.assert_called_once()
        _, kwargs = mock_client.set.call_args
        props = kwargs.get('properties', {})
        assert props['email'] == 'alice@example.com'
        assert props['org_id'] == 'org-42'
        assert props['org_name'] == 'Acme Corp'
        assert props['idp'] == 'github'
        assert 'last_login_at' in props
        assert 'plan_tier' in props

    def test_identify_user_saas_with_orgs_calls_group_identify(self, saas_service):
        """identify_user in SaaS mode with orgs calls group_identify for each org."""
        service, mock_client = saas_service
        orgs = [
            {'id': 'org-1', 'name': 'Org One', 'member_count': 5},
            {'id': 'org-2', 'name': 'Org Two', 'member_count': 10},
        ]
        ctx = make_ctx(user_id='user-1')
        service.identify_user(
            ctx=ctx,
            orgs=orgs,
        )
        assert mock_client.group_identify.call_count == 2

    def test_identify_user_with_user_none_skips_all(self, saas_service):
        """identify_user with no email/org/orgs still calls set_person_properties (with None values)."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.identify_user(
            ctx=ctx,
        )
        # set_person_properties is still called (with None fields)
        mock_client.set.assert_called_once()
        # but group_identify is NOT called since no orgs provided
        mock_client.group_identify.assert_not_called()

    def test_identify_user_catches_exception(self, saas_service):
        """identify_user catches any exception internally and does not raise."""
        service, mock_client = saas_service
        mock_client.set.side_effect = RuntimeError('PostHog SDK error')
        ctx = make_ctx(user_id='user-1')
        # Should not raise
        service.identify_user(
            ctx=ctx,
            email='a@b.com',
        )

    def test_identify_user_org_with_no_name(self, saas_service):
        """identify_user with org that has no name handles it gracefully."""
        service, mock_client = saas_service
        orgs = [
            {'id': 'org-1', 'name': None, 'member_count': 3},
        ]
        ctx = make_ctx(user_id='user-1')
        service.identify_user(
            ctx=ctx,
            orgs=orgs,
        )
        assert mock_client.group_identify.call_count == 1
        _, kwargs = mock_client.group_identify.call_args
        props = kwargs.get('properties', {})
        assert props.get('org_name') is None


# ---------------------------------------------------------------------------
# Typed event methods tests
# ---------------------------------------------------------------------------


class TestTypedEventMethods:
    """Tests for the 15 typed event methods on AnalyticsService."""

    def test_track_user_signed_up(self, saas_service):
        """track_user_signed_up calls capture with USER_SIGNED_UP and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_user_signed_up(
            ctx=ctx,
            email_domain='example.com',
            invitation_source='invite_link',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == USER_SIGNED_UP
        props = kwargs['properties']
        assert props['email_domain'] == 'example.com'
        assert props['invitation_source'] == 'invite_link'

    def test_track_user_logged_in(self, saas_service):
        """track_user_logged_in calls capture with USER_LOGGED_IN and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_user_logged_in(
            ctx=ctx,
            idp='google',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == USER_LOGGED_IN
        props = kwargs['properties']
        assert props['idp'] == 'google'

    def test_track_conversation_created(self, saas_service):
        """track_conversation_created calls capture with CONVERSATION_CREATED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_conversation_created(
            ctx=ctx,
            conversation_id='conv-abc',
            trigger='ui',
            llm_model='gpt-4',
            agent_type='CodeActAgent',
            has_repository=True,
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == CONVERSATION_CREATED
        props = kwargs['properties']
        assert props['conversation_id'] == 'conv-abc'
        assert props['trigger'] == 'ui'
        assert props['llm_model'] == 'gpt-4'
        assert props['agent_type'] == 'CodeActAgent'
        assert props['has_repository'] is True

    def test_track_conversation_finished(self, saas_service):
        """track_conversation_finished calls capture with CONVERSATION_FINISHED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_conversation_finished(
            ctx=ctx,
            conversation_id='conv-abc',
            terminal_state='completed',
            turn_count=5,
            accumulated_cost_usd=0.15,
            prompt_tokens=1000,
            completion_tokens=500,
            llm_model='gpt-4',
            trigger='ui',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == CONVERSATION_FINISHED
        props = kwargs['properties']
        assert props['conversation_id'] == 'conv-abc'
        assert props['terminal_state'] == 'completed'
        assert props['turn_count'] == 5
        assert props['accumulated_cost_usd'] == 0.15
        assert props['prompt_tokens'] == 1000
        assert props['completion_tokens'] == 500
        assert props['llm_model'] == 'gpt-4'
        assert props['trigger'] == 'ui'

    def test_track_conversation_errored(self, saas_service):
        """track_conversation_errored calls capture with CONVERSATION_ERRORED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_conversation_errored(
            ctx=ctx,
            conversation_id='conv-abc',
            error_type='LLMError',
            error_message='Rate limit exceeded',
            llm_model='gpt-4',
            turn_count=3,
            terminal_state='error',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == CONVERSATION_ERRORED
        props = kwargs['properties']
        assert props['conversation_id'] == 'conv-abc'
        assert props['error_type'] == 'LLMError'
        assert props['error_message'] == 'Rate limit exceeded'
        assert props['llm_model'] == 'gpt-4'
        assert props['turn_count'] == 3
        assert props['terminal_state'] == 'error'

    def test_track_credit_purchased(self, saas_service):
        """track_credit_purchased calls capture with CREDIT_PURCHASED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_credit_purchased(
            ctx=ctx,
            amount_usd=10.0,
            credit_balance_before=5.0,
            credit_balance_after=15.0,
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == CREDIT_PURCHASED
        props = kwargs['properties']
        assert props['amount_usd'] == 10.0
        assert props['credit_balance_before'] == 5.0
        assert props['credit_balance_after'] == 15.0

    def test_track_credit_limit_reached(self, saas_service):
        """track_credit_limit_reached calls capture with CREDIT_LIMIT_REACHED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_credit_limit_reached(
            ctx=ctx,
            conversation_id='conv-abc',
            credit_balance=0.0,
            llm_model='gpt-4',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == CREDIT_LIMIT_REACHED
        props = kwargs['properties']
        assert props['conversation_id'] == 'conv-abc'
        assert props['credit_balance'] == 0.0
        assert props['llm_model'] == 'gpt-4'

    def test_track_git_provider_connected(self, saas_service):
        """track_git_provider_connected calls capture with GIT_PROVIDER_CONNECTED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_git_provider_connected(
            ctx=ctx,
            provider_type='github',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == GIT_PROVIDER_CONNECTED
        props = kwargs['properties']
        assert props['provider_type'] == 'github'

    def test_track_onboarding_completed(self, saas_service):
        """track_onboarding_completed calls capture with ONBOARDING_COMPLETED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_onboarding_completed(
            ctx=ctx,
            selections={
                'role': 'developer',
                'org_size': '11-50',
                'use_case': 'code_review',
            },
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == ONBOARDING_COMPLETED
        props = kwargs['properties']
        assert props['role'] == 'developer'
        assert props['org_size'] == '11-50'
        assert props['use_case'] == 'code_review'

    def test_track_conversation_deleted(self, saas_service):
        """track_conversation_deleted calls capture with CONVERSATION_DELETED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_conversation_deleted(
            ctx=ctx,
            conversation_id='conv-xyz',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == CONVERSATION_DELETED
        props = kwargs['properties']
        assert props['conversation_id'] == 'conv-xyz'

    def test_track_settings_saved(self, saas_service):
        """track_settings_saved calls capture with SETTINGS_SAVED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_settings_saved(
            ctx=ctx,
            settings_changed=['llm_model', 'agent_type'],
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == SETTINGS_SAVED
        props = kwargs['properties']
        assert props['settings_changed'] == ['llm_model', 'agent_type']

    def test_track_trajectory_downloaded(self, saas_service):
        """track_trajectory_downloaded calls capture with TRAJECTORY_DOWNLOADED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1')
        service.track_trajectory_downloaded(
            ctx=ctx,
            conversation_id='conv-download-123',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == TRAJECTORY_DOWNLOADED
        props = kwargs['properties']
        assert props['conversation_id'] == 'conv-download-123'

    def test_track_team_members_invited(self, saas_service):
        """track_team_members_invited calls capture with TEAM_MEMBERS_INVITED and correct properties."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1', org_id='org-123')
        service.track_team_members_invited(
            ctx=ctx,
            invited_count=5,
            successful_count=4,
            failed_count=1,
            role='member',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        assert kwargs['event'] == TEAM_MEMBERS_INVITED
        props = kwargs['properties']
        assert props['invited_count'] == 5
        assert props['successful_count'] == 4
        assert props['failed_count'] == 1
        assert props['role'] == 'member'

    def test_typed_method_consent_false_is_noop(self, saas_service):
        """A typed method with consented=False results in no capture call."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1', consented=False)
        service.track_user_logged_in(
            ctx=ctx,
            idp='github',
        )
        mock_client.capture.assert_not_called()

    def test_typed_method_passes_org_id(self, saas_service):
        """A typed method passes org_id through to self.capture."""
        service, mock_client = saas_service
        ctx = make_ctx(user_id='user-1', org_id='org-99')
        service.track_user_logged_in(
            ctx=ctx,
            idp='github',
        )
        mock_client.capture.assert_called_once()
        _, kwargs = mock_client.capture.call_args
        props = kwargs['properties']
        assert props.get('org_id') == 'org-99'
