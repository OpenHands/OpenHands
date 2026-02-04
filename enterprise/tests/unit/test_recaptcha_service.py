"""Tests for RecaptchaService."""

import hashlib
import hmac
from unittest.mock import MagicMock, patch

import pytest
from server.auth.recaptcha_service import AssessmentResult, RecaptchaService


@pytest.fixture
def mock_gcp_client():
    """Mock GCP reCAPTCHA Enterprise client."""
    with patch(
        'server.auth.recaptcha_service.recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient'
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def recaptcha_service(mock_gcp_client):
    """Create RecaptchaService instance with mocked dependencies."""
    with (
        patch('server.auth.recaptcha_service.RECAPTCHA_PROJECT_ID', 'test-project'),
        patch('server.auth.recaptcha_service.RECAPTCHA_SITE_KEY', 'test-site-key'),
        patch('server.auth.recaptcha_service.RECAPTCHA_HMAC_SECRET', 'test-secret'),
        patch('server.auth.recaptcha_service.RECAPTCHA_BLOCK_THRESHOLD', 0.3),
    ):
        # Create new instance - constants are imported at module level, so we patch the imported names
        return RecaptchaService()


class TestRecaptchaServiceHashAccountId:
    """Tests for RecaptchaService.hash_account_id()."""

    def test_should_hash_email_with_hmac_sha256(self, recaptcha_service):
        """Test that hash_account_id produces correct HMAC-SHA256 hash."""
        # Arrange
        email = 'user@example.com'
        # The service reads RECAPTCHA_HMAC_SECRET from the imported constants
        # We need to verify it uses the constant correctly
        from server.auth.recaptcha_service import RECAPTCHA_HMAC_SECRET

        # Act
        result = recaptcha_service.hash_account_id(email)

        # Assert
        # Verify the hash is correct using the actual secret from the patched constant
        expected_hash = hmac.new(
            RECAPTCHA_HMAC_SECRET.encode(),
            email.lower().encode(),
            hashlib.sha256,
        ).hexdigest()
        assert result == expected_hash
        assert len(result) == 64  # SHA256 produces 64 hex characters

    def test_should_normalize_email_to_lowercase(self, recaptcha_service):
        """Test that hash_account_id normalizes email to lowercase."""
        # Arrange
        email1 = 'User@Example.com'
        email2 = 'user@example.com'

        # Act
        hash1 = recaptcha_service.hash_account_id(email1)
        hash2 = recaptcha_service.hash_account_id(email2)

        # Assert
        assert hash1 == hash2

    def test_should_produce_different_hashes_for_different_emails(
        self, recaptcha_service
    ):
        """Test that different emails produce different hashes."""
        # Arrange
        email1 = 'user1@example.com'
        email2 = 'user2@example.com'

        # Act
        hash1 = recaptcha_service.hash_account_id(email1)
        hash2 = recaptcha_service.hash_account_id(email2)

        # Assert
        assert hash1 != hash2


class TestRecaptchaServiceCreateAssessment:
    """Tests for RecaptchaService.create_assessment()."""

    def test_should_create_assessment_and_allow_when_score_is_high(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that assessment allows request when score is above threshold."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/abc123'
        mock_response.token_properties.valid = True
        mock_response.token_properties.action = 'LOGIN'
        mock_response.risk_analysis.score = 0.9
        mock_response.risk_analysis.reasons = []
        mock_gcp_client.create_assessment.return_value = mock_response

        # Act
        result = recaptcha_service.create_assessment(
            token='test-token',
            action='LOGIN',
            user_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
        )

        # Assert
        assert isinstance(result, AssessmentResult)
        assert result.name == 'projects/test-project/assessments/abc123'
        assert result.allowed is True
        assert result.score == 0.9
        assert result.valid is True
        assert result.action_valid is True
        mock_gcp_client.create_assessment.assert_called_once()

    def test_should_block_when_score_is_below_threshold(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that assessment blocks request when score is below threshold."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/def456'
        mock_response.token_properties.valid = True
        mock_response.token_properties.action = 'LOGIN'
        mock_response.risk_analysis.score = 0.2
        mock_response.risk_analysis.reasons = []
        mock_gcp_client.create_assessment.return_value = mock_response

        # Act
        result = recaptcha_service.create_assessment(
            token='test-token',
            action='LOGIN',
            user_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
        )

        # Assert
        assert result.allowed is False
        assert result.score == 0.2

    def test_should_block_when_token_is_invalid(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that assessment blocks request when token is invalid."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/ghi789'
        mock_response.token_properties.valid = False
        mock_response.token_properties.action = 'LOGIN'
        mock_response.risk_analysis.score = 0.9
        mock_response.risk_analysis.reasons = []
        mock_gcp_client.create_assessment.return_value = mock_response

        # Act
        result = recaptcha_service.create_assessment(
            token='invalid-token',
            action='LOGIN',
            user_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
        )

        # Assert
        assert result.allowed is False
        assert result.valid is False

    def test_should_block_when_action_does_not_match(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that assessment blocks request when action doesn't match."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/jkl012'
        mock_response.token_properties.valid = True
        mock_response.token_properties.action = 'SIGNUP'
        mock_response.risk_analysis.score = 0.9
        mock_response.risk_analysis.reasons = []
        mock_gcp_client.create_assessment.return_value = mock_response

        # Act
        result = recaptcha_service.create_assessment(
            token='test-token',
            action='LOGIN',
            user_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
        )

        # Assert
        assert result.allowed is False
        assert result.action_valid is False

    def test_should_include_email_in_user_info_when_provided(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that email is included in user_info when provided."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/mno345'
        mock_response.token_properties.valid = True
        mock_response.token_properties.action = 'LOGIN'
        mock_response.risk_analysis.score = 0.9
        mock_response.risk_analysis.reasons = []
        mock_gcp_client.create_assessment.return_value = mock_response

        # Act
        recaptcha_service.create_assessment(
            token='test-token',
            action='LOGIN',
            user_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
            email='user@example.com',
        )

        # Assert
        call_args = mock_gcp_client.create_assessment.call_args
        assessment = call_args[0][0].assessment
        assert assessment.event.user_info is not None
        assert assessment.event.user_info.account_id is not None
        assert len(assessment.event.user_info.user_ids) == 1
        assert assessment.event.user_info.user_ids[0].email == 'user@example.com'

    def test_should_not_include_user_info_when_email_is_none(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that user_info is not included when email is None."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/pqr678'
        mock_response.token_properties.valid = True
        mock_response.token_properties.action = 'LOGIN'
        mock_response.risk_analysis.score = 0.9
        mock_response.risk_analysis.reasons = []
        mock_gcp_client.create_assessment.return_value = mock_response

        # Act
        recaptcha_service.create_assessment(
            token='test-token',
            action='LOGIN',
            user_ip='192.168.1.1',
            user_agent='Mozilla/5.0',
            email=None,
        )

        # Assert
        call_args = mock_gcp_client.create_assessment.call_args
        assessment = call_args[0][0].assessment
        # When email is None, user_info should not be set
        # Check that user_info was not explicitly set (protobuf objects may have default empty values)
        # The key is that account_id should not be set when email is None
        if hasattr(assessment.event, 'user_info') and assessment.event.user_info:
            # If user_info exists, verify account_id is empty (not set)
            assert not assessment.event.user_info.account_id

    def test_should_log_assessment_details_including_name(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that assessment details including assessment name are logged."""
        # Arrange
        mock_response = MagicMock()
        mock_response.name = 'projects/test-project/assessments/stu901'
        mock_response.token_properties.valid = True
        mock_response.token_properties.action = 'LOGIN'
        mock_response.risk_analysis.score = 0.9
        mock_response.risk_analysis.reasons = ['AUTOMATION']
        mock_gcp_client.create_assessment.return_value = mock_response

        with patch('server.auth.recaptcha_service.logger') as mock_logger:
            # Act
            recaptcha_service.create_assessment(
                token='test-token',
                action='LOGIN',
                user_ip='192.168.1.1',
                user_agent='Mozilla/5.0',
            )

            # Assert
            mock_logger.info.assert_called_once()
            call_kwargs = mock_logger.info.call_args
            assert call_kwargs[0][0] == 'recaptcha_assessment'
            assert (
                call_kwargs[1]['extra']['assessment_name']
                == 'projects/test-project/assessments/stu901'
            )
            assert call_kwargs[1]['extra']['score'] == 0.9
            assert call_kwargs[1]['extra']['valid'] is True
            assert call_kwargs[1]['extra']['action_valid'] is True
            assert call_kwargs[1]['extra']['user_ip'] == '192.168.1.1'

    def test_should_raise_exception_when_gcp_client_fails(
        self, recaptcha_service, mock_gcp_client
    ):
        """Test that exceptions from GCP client are propagated."""
        # Arrange
        mock_gcp_client.create_assessment.side_effect = Exception('GCP error')

        # Act & Assert
        with pytest.raises(Exception, match='GCP error'):
            recaptcha_service.create_assessment(
                token='test-token',
                action='LOGIN',
                user_ip='192.168.1.1',
                user_agent='Mozilla/5.0',
            )
