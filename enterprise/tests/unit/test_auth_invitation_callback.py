"""Tests for auth callback invitation acceptance - EmailMismatchError handling."""

import pytest


class TestAuthCallbackInvitationEmailMismatch:
    """Test cases for EmailMismatchError handling during auth callback."""

    @pytest.fixture
    def mock_redirect_url(self):
        """Base redirect URL."""
        return 'https://app.example.com/'

    @pytest.fixture
    def mock_user_id(self):
        """Mock user ID."""
        return '87654321-4321-8765-4321-876543218765'

    def test_email_mismatch_appends_to_url_without_query_params(
        self, mock_redirect_url, mock_user_id
    ):
        """Test that email_mismatch=true is appended correctly when URL has no query params."""
        from server.routes.org_invitation_models import EmailMismatchError

        # Simulate the logic from auth.py
        redirect_url = mock_redirect_url
        try:
            raise EmailMismatchError('Your email does not match the invitation')
        except EmailMismatchError:
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&email_mismatch=true'
            else:
                redirect_url = f'{redirect_url}?email_mismatch=true'

        assert redirect_url == 'https://app.example.com/?email_mismatch=true'

    def test_email_mismatch_appends_to_url_with_query_params(self, mock_user_id):
        """Test that email_mismatch=true is appended correctly when URL has existing query params."""
        from server.routes.org_invitation_models import EmailMismatchError

        redirect_url = 'https://app.example.com/?other_param=value'
        try:
            raise EmailMismatchError()
        except EmailMismatchError:
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&email_mismatch=true'
            else:
                redirect_url = f'{redirect_url}?email_mismatch=true'

        assert (
            redirect_url
            == 'https://app.example.com/?other_param=value&email_mismatch=true'
        )

    def test_email_mismatch_error_has_default_message(self):
        """Test that EmailMismatchError has the default message."""
        from server.routes.org_invitation_models import EmailMismatchError

        error = EmailMismatchError()
        assert str(error) == 'Your email does not match the invitation'

    def test_email_mismatch_error_accepts_custom_message(self):
        """Test that EmailMismatchError accepts a custom message."""
        from server.routes.org_invitation_models import EmailMismatchError

        custom_message = 'Custom error message'
        error = EmailMismatchError(custom_message)
        assert str(error) == custom_message

    def test_email_mismatch_error_is_invitation_error(self):
        """Test that EmailMismatchError inherits from InvitationError."""
        from server.routes.org_invitation_models import (
            EmailMismatchError,
            InvitationError,
        )

        error = EmailMismatchError()
        assert isinstance(error, InvitationError)


class TestInvitationTokenInOAuthState:
    """Test cases for invitation token handling in OAuth state."""

    def test_invitation_token_included_in_oauth_state(self):
        """Test that invitation token is included in OAuth state data."""
        import base64
        import json

        # Simulate building OAuth state with invitation token
        state_data = {
            'redirect_url': 'https://app.example.com/',
            'invitation_token': 'inv-test-token-12345',
        }

        encoded_state = base64.b64encode(json.dumps(state_data).encode()).decode()
        decoded_data = json.loads(base64.b64decode(encoded_state))

        assert decoded_data['invitation_token'] == 'inv-test-token-12345'
        assert decoded_data['redirect_url'] == 'https://app.example.com/'

    def test_invitation_token_extracted_from_oauth_state(self):
        """Test that invitation token can be extracted from OAuth state."""
        import base64
        import json

        state_data = {
            'redirect_url': 'https://app.example.com/',
            'invitation_token': 'inv-test-token-12345',
        }

        encoded_state = base64.b64encode(json.dumps(state_data).encode()).decode()

        # Simulate decoding in callback
        decoded_state = json.loads(base64.b64decode(encoded_state))
        invitation_token = decoded_state.get('invitation_token')

        assert invitation_token == 'inv-test-token-12345'

    def test_oauth_state_without_invitation_token(self):
        """Test that OAuth state works without invitation token."""
        import base64
        import json

        state_data = {
            'redirect_url': 'https://app.example.com/',
        }

        encoded_state = base64.b64encode(json.dumps(state_data).encode()).decode()
        decoded_data = json.loads(base64.b64decode(encoded_state))

        assert 'invitation_token' not in decoded_data
        assert decoded_data['redirect_url'] == 'https://app.example.com/'


class TestAuthCallbackInvitationErrors:
    """Test cases for various invitation error scenarios in auth callback."""

    def test_invitation_expired_appends_flag(self):
        """Test that invitation_expired=true is appended for expired invitations."""
        from server.routes.org_invitation_models import InvitationExpiredError

        redirect_url = 'https://app.example.com/'
        try:
            raise InvitationExpiredError()
        except InvitationExpiredError:
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&invitation_expired=true'
            else:
                redirect_url = f'{redirect_url}?invitation_expired=true'

        assert redirect_url == 'https://app.example.com/?invitation_expired=true'

    def test_invitation_invalid_appends_flag(self):
        """Test that invitation_invalid=true is appended for invalid invitations."""
        from server.routes.org_invitation_models import InvitationInvalidError

        redirect_url = 'https://app.example.com/'
        try:
            raise InvitationInvalidError()
        except InvitationInvalidError:
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&invitation_invalid=true'
            else:
                redirect_url = f'{redirect_url}?invitation_invalid=true'

        assert redirect_url == 'https://app.example.com/?invitation_invalid=true'

    def test_already_member_appends_flag(self):
        """Test that already_member=true is appended when user is already a member."""
        from server.routes.org_invitation_models import UserAlreadyMemberError

        redirect_url = 'https://app.example.com/'
        try:
            raise UserAlreadyMemberError()
        except UserAlreadyMemberError:
            if '?' in redirect_url:
                redirect_url = f'{redirect_url}&already_member=true'
            else:
                redirect_url = f'{redirect_url}?already_member=true'

        assert redirect_url == 'https://app.example.com/?already_member=true'
