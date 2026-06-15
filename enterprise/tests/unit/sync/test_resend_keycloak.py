"""Tests for Resend Keycloak sync functionality."""

import os
from unittest.mock import MagicMock, call, patch
from uuid import UUID

import pytest
from resend.exceptions import ResendError
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from tenacity import RetryError

# Set required environment variables before importing the module
# that reads them at import time
os.environ['RESEND_API_KEY'] = 'test_api_key'
os.environ['RESEND_AUDIENCE_ID'] = 'test_audience_id'

from enterprise.sync.resend_keycloak import (  # noqa: E402
    BATCH_SIZE,
    ResendUser,
    _split_display_name,
    add_contact_to_resend,
    get_local_users,
    get_total_local_users,
    is_valid_email,
    send_welcome_email,
    sync_users_to_resend,
)


class TestIsValidEmail:
    """Test cases for is_valid_email function."""

    def test_valid_simple_email(self):
        """Test that a simple valid email passes validation."""
        assert is_valid_email('user@example.com') is True

    def test_valid_email_with_plus(self):
        """Test that email with + modifier passes validation."""
        assert is_valid_email('user+tag@example.com') is True

    def test_valid_email_with_dots(self):
        """Test that email with dots in local part passes validation."""
        assert is_valid_email('first.last@example.com') is True

    def test_valid_email_with_numbers(self):
        """Test that email with numbers passes validation."""
        assert is_valid_email('user123@example.com') is True

    def test_valid_email_with_subdomain(self):
        """Test that email with subdomain passes validation."""
        assert is_valid_email('user@mail.example.com') is True

    def test_valid_email_with_hyphen_domain(self):
        """Test that email with hyphen in domain passes validation."""
        assert is_valid_email('user@example-site.com') is True

    def test_valid_email_with_underscore(self):
        """Test that email with underscore passes validation."""
        assert is_valid_email('user_name@example.com') is True

    def test_valid_email_with_percent(self):
        """Test that email with percent sign passes validation."""
        assert is_valid_email('user%name@example.com') is True

    def test_invalid_email_with_exclamation(self):
        """Test that email with exclamation mark fails validation.

        This is the specific case from the bug report:
        ethanjames3713+!@gmail.com
        """
        assert is_valid_email('ethanjames3713+!@gmail.com') is False

    def test_invalid_email_with_special_chars(self):
        """Test that email with other special characters fails validation."""
        assert is_valid_email('user!name@example.com') is False
        assert is_valid_email('user#name@example.com') is False
        assert is_valid_email('user$name@example.com') is False
        assert is_valid_email('user&name@example.com') is False
        assert is_valid_email("user'name@example.com") is False
        assert is_valid_email('user*name@example.com') is False
        assert is_valid_email('user=name@example.com') is False
        assert is_valid_email('user^name@example.com') is False
        assert is_valid_email('user`name@example.com') is False
        assert is_valid_email('user{name@example.com') is False
        assert is_valid_email('user|name@example.com') is False
        assert is_valid_email('user}name@example.com') is False
        assert is_valid_email('user~name@example.com') is False

    def test_invalid_email_no_at_symbol(self):
        """Test that email without @ symbol fails validation."""
        assert is_valid_email('userexample.com') is False

    def test_invalid_email_no_domain(self):
        """Test that email without domain fails validation."""
        assert is_valid_email('user@') is False

    def test_invalid_email_no_local_part(self):
        """Test that email without local part fails validation."""
        assert is_valid_email('@example.com') is False

    def test_invalid_email_no_tld(self):
        """Test that email without TLD fails validation."""
        assert is_valid_email('user@example') is False

    def test_invalid_email_single_char_tld(self):
        """Test that email with single character TLD fails validation."""
        assert is_valid_email('user@example.c') is False

    def test_invalid_email_empty_string(self):
        """Test that empty string fails validation."""
        assert is_valid_email('') is False

    def test_invalid_email_none(self):
        """Test that None fails validation."""
        assert is_valid_email(None) is False

    def test_invalid_email_whitespace(self):
        """Test that email with whitespace fails validation."""
        assert is_valid_email('user @example.com') is False
        assert is_valid_email('user@ example.com') is False
        assert is_valid_email(' user@example.com') is False
        assert is_valid_email('user@example.com ') is False

    def test_invalid_email_double_at(self):
        """Test that email with double @ fails validation."""
        assert is_valid_email('user@@example.com') is False

    def test_email_double_dot_domain(self):
        """Test email with double dot in domain.

        Note: The regex allows this as it's technically valid in some edge cases,
        and Resend's API may accept it. The main goal is to reject special
        characters like ! that Resend definitely rejects.
        """
        # This is allowed by our regex - Resend may or may not accept it
        assert is_valid_email('user@example..com') is True

    def test_case_insensitive_validation(self):
        """Test that validation works for uppercase emails."""
        assert is_valid_email('USER@EXAMPLE.COM') is True
        assert is_valid_email('User@Example.Com') is True


class TestDisplayNameParsing:
    def test_split_display_name_handles_full_single_and_blank_names(self) -> None:
        assert _split_display_name('Ada Lovelace') == ('Ada', 'Lovelace')
        assert _split_display_name('Prince') == ('Prince', None)
        assert _split_display_name('  Grace Brewster Hopper  ') == (
            'Grace',
            'Brewster Hopper',
        )
        assert _split_display_name('') == (None, None)
        assert _split_display_name(None) == (None, None)


class _LocalUserBase(DeclarativeBase):
    pass


class _LocalUser(_LocalUserBase):
    __tablename__ = 'user'

    id: Mapped[UUID] = mapped_column(primary_key=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    git_user_name: Mapped[str | None] = mapped_column(String, nullable=True)


def _local_user_session_maker():
    engine = create_engine('sqlite:///:memory:')
    _LocalUserBase.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        session.add_all(
            [
                _LocalUser(
                    id=UUID('00000000-0000-0000-0000-000000000001'),
                    email='ada@example.com',
                    git_user_name='Ada Lovelace',
                ),
                _LocalUser(
                    id=UUID('00000000-0000-0000-0000-000000000002'),
                    email=None,
                    git_user_name='No Email',
                ),
                _LocalUser(
                    id=UUID('00000000-0000-0000-0000-000000000003'),
                    email='',
                    git_user_name='Blank Email',
                ),
                _LocalUser(
                    id=UUID('00000000-0000-0000-0000-000000000004'),
                    email='prince@example.com',
                    git_user_name='Prince',
                ),
            ]
        )
        session.commit()

    return session_factory


class TestLocalUserQueries:
    @patch('enterprise.sync.resend_keycloak.User', _LocalUser)
    @patch('enterprise.sync.resend_keycloak._get_session_maker')
    def test_get_local_users_reads_real_database_with_names(
        self, mock_get_session_maker: MagicMock
    ) -> None:
        mock_get_session_maker.return_value = _local_user_session_maker()

        users = get_local_users(offset=0, limit=10)

        assert users == [
            ResendUser(
                id='00000000-0000-0000-0000-000000000001',
                email='ada@example.com',
                first_name='Ada',
                last_name='Lovelace',
            ),
            ResendUser(
                id='00000000-0000-0000-0000-000000000004',
                email='prince@example.com',
                first_name='Prince',
                last_name=None,
            ),
        ]

    @patch('enterprise.sync.resend_keycloak.User', _LocalUser)
    @patch('enterprise.sync.resend_keycloak._get_session_maker')
    def test_get_total_local_users_counts_real_database_emails(
        self, mock_get_session_maker: MagicMock
    ) -> None:
        mock_get_session_maker.return_value = _local_user_session_maker()

        assert get_total_local_users() == 2


class TestSendWelcomeEmail:
    """Tests for send_welcome_email function."""

    @patch('enterprise.sync.resend_keycloak.resend.Emails.send')
    def test_send_welcome_email_success(self, mock_send: MagicMock) -> None:
        """Test successful welcome email sending."""
        mock_send.return_value = {'id': 'email_123'}

        result = send_welcome_email(
            email='test@example.com',
            first_name='John',
            last_name='Doe',
        )

        assert result == {'id': 'email_123'}
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert call_args['to'] == ['test@example.com']
        assert call_args['subject'] == 'Welcome to OpenHands Cloud'
        assert 'Hi John Doe,' in call_args['html']

    @patch('enterprise.sync.resend_keycloak.resend.Emails.send')
    def test_send_welcome_email_retries_on_rate_limit(
        self, mock_send: MagicMock
    ) -> None:
        """Test that send_welcome_email retries on rate limit errors."""
        # First two calls raise rate limit error, third succeeds
        mock_send.side_effect = [
            ResendError(
                code=429,
                message='Too many requests',
                error_type='rate_limit_exceeded',
                suggested_action='',
            ),
            ResendError(
                code=429,
                message='Too many requests',
                error_type='rate_limit_exceeded',
                suggested_action='',
            ),
            {'id': 'email_123'},
        ]

        result = send_welcome_email(
            email='test@example.com',
            first_name='John',
            last_name='Doe',
        )

        assert result == {'id': 'email_123'}
        assert mock_send.call_count == 3

    @patch('enterprise.sync.resend_keycloak.resend.Emails.send')
    def test_send_welcome_email_fails_after_max_retries(
        self, mock_send: MagicMock
    ) -> None:
        """Test that send_welcome_email fails after max retries."""
        # All calls raise rate limit error
        mock_send.side_effect = ResendError(
            code=429,
            message='Too many requests',
            error_type='rate_limit_exceeded',
            suggested_action='',
        )

        # Tenacity wraps the final exception in RetryError
        with pytest.raises(RetryError):
            send_welcome_email(
                email='test@example.com',
                first_name='John',
                last_name='Doe',
            )

        # Default MAX_RETRIES is 3
        assert mock_send.call_count == 3

    @patch('enterprise.sync.resend_keycloak.resend.Emails.send')
    def test_send_welcome_email_no_name(self, mock_send: MagicMock) -> None:
        """Test welcome email with no name provided."""
        mock_send.return_value = {'id': 'email_123'}

        result = send_welcome_email(email='test@example.com')

        assert result == {'id': 'email_123'}
        call_args = mock_send.call_args[0][0]
        assert 'Hi there,' in call_args['html']


class TestAddContactToResend:
    """Tests for add_contact_to_resend function."""

    @patch('enterprise.sync.resend_keycloak.resend.Contacts.create')
    def test_add_contact_to_resend_success(self, mock_create: MagicMock) -> None:
        """Test successful contact addition."""
        mock_create.return_value = {'id': 'contact_123'}

        result = add_contact_to_resend(
            audience_id='test_audience',
            email='test@example.com',
            first_name='John',
            last_name='Doe',
        )

        assert result == {'id': 'contact_123'}
        mock_create.assert_called_once()

    @patch('enterprise.sync.resend_keycloak.resend.Contacts.create')
    def test_add_contact_to_resend_retries_on_rate_limit(
        self, mock_create: MagicMock
    ) -> None:
        """Test that add_contact_to_resend retries on rate limit errors."""
        # First call raises rate limit error, second succeeds
        mock_create.side_effect = [
            ResendError(
                code=429,
                message='Too many requests',
                error_type='rate_limit_exceeded',
                suggested_action='',
            ),
            {'id': 'contact_123'},
        ]

        result = add_contact_to_resend(
            audience_id='test_audience',
            email='test@example.com',
        )

        assert result == {'id': 'contact_123'}
        assert mock_create.call_count == 2


class TestSyncUsersToResend:
    @patch('enterprise.sync.resend_keycloak.time.sleep')
    @patch('enterprise.sync.resend_keycloak.send_welcome_email')
    @patch('enterprise.sync.resend_keycloak.add_contact_to_resend')
    @patch('enterprise.sync.resend_keycloak.get_local_users')
    @patch('enterprise.sync.resend_keycloak.get_total_local_users')
    @patch('enterprise.sync.resend_keycloak._backfill_existing_resend_contacts')
    @patch('enterprise.sync.resend_keycloak._get_resend_synced_user_store')
    def test_sync_reads_local_users_and_skips_synced_or_invalid_emails(
        self,
        mock_get_store: MagicMock,
        mock_backfill: MagicMock,
        mock_get_total: MagicMock,
        mock_get_local_users: MagicMock,
        mock_add_contact: MagicMock,
        mock_send_welcome: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        store = MagicMock()
        store.get_synced_emails_for_audience.return_value = {'already@example.com'}
        mock_get_store.return_value = store
        mock_backfill.return_value = 1
        mock_get_total.return_value = 3
        mock_get_local_users.return_value = [
            ResendUser(id='user-1', email='already@example.com'),
            ResendUser(id='user-2', email='bad!email@example.com'),
            ResendUser(
                id='user-3',
                email='new@example.com',
                first_name='Ada',
                last_name='Lovelace',
            ),
        ]

        sync_users_to_resend()

        mock_get_total.assert_called_once_with()
        mock_get_local_users.assert_called_once_with(0, BATCH_SIZE)
        store.mark_user_synced.assert_called_once_with(
            email='new@example.com',
            audience_id='test_audience_id',
            user_id='user-3',
        )
        mock_add_contact.assert_called_once_with(
            'test_audience_id', 'new@example.com', 'Ada', 'Lovelace'
        )
        mock_send_welcome.assert_called_once_with('new@example.com', 'Ada', 'Lovelace')
        assert mock_sleep.call_count == 2

    @patch('enterprise.sync.resend_keycloak.time.sleep')
    @patch('enterprise.sync.resend_keycloak.send_welcome_email')
    @patch('enterprise.sync.resend_keycloak.add_contact_to_resend')
    @patch('enterprise.sync.resend_keycloak.get_local_users')
    @patch('enterprise.sync.resend_keycloak.get_total_local_users')
    @patch('enterprise.sync.resend_keycloak._backfill_existing_resend_contacts')
    @patch('enterprise.sync.resend_keycloak._get_resend_synced_user_store')
    def test_sync_reads_multiple_local_user_batches(
        self,
        mock_get_store: MagicMock,
        mock_backfill: MagicMock,
        mock_get_total: MagicMock,
        mock_get_local_users: MagicMock,
        mock_add_contact: MagicMock,
        mock_send_welcome: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        store = MagicMock()
        store.get_synced_emails_for_audience.return_value = set()
        mock_get_store.return_value = store
        mock_backfill.return_value = 0
        mock_get_total.return_value = BATCH_SIZE + 1
        mock_get_local_users.side_effect = [
            [ResendUser(id='user-1', email='new@example.com')],
            [],
        ]

        sync_users_to_resend()

        mock_get_total.assert_called_once_with()
        mock_get_local_users.assert_has_calls(
            [call(0, BATCH_SIZE), call(BATCH_SIZE, BATCH_SIZE)]
        )
        assert mock_get_local_users.call_count == 2
        store.mark_user_synced.assert_called_once_with(
            email='new@example.com',
            audience_id='test_audience_id',
            user_id='user-1',
        )
        mock_add_contact.assert_called_once_with(
            'test_audience_id', 'new@example.com', None, None
        )
        mock_send_welcome.assert_called_once_with('new@example.com', None, None)
        assert mock_sleep.call_count == 2
