"""
Tests for GitLabService conditional token assignment.

This module tests that GitLabService only assigns the token parameter to self.token
when external_token_manager is False, preventing static tokens from bypassing
dynamic token fetching.
"""

from pydantic import SecretStr

from openhands.integrations.gitlab.gitlab_service import GitLabService


class TestGitLabServiceTokenConditionalAssignment:
    """Test suite for GitLabService conditional token assignment."""

    def test_assigns_token_when_external_manager_disabled(self):
        """
        Test: Token is assigned when external_token_manager is False.

        Arrange: Create GitLabService with token and external_token_manager=False
        Act: Initialize the service
        Assert: Token is assigned to self.token
        """
        # Arrange
        token = 'test_token_abc123'

        # Act
        service = GitLabService(
            token=token,
            external_token_manager=False,
        )

        # Assert
        # Token is stored as a string, not SecretStr in the base class
        assert service.token == token

    def test_does_not_assign_token_when_external_manager_enabled(self):
        """
        Test: Token is not assigned when external_token_manager is True.

        Arrange: Create GitLabService with token and external_token_manager=True
        Act: Initialize the service
        Assert: Token is not assigned (remains empty SecretStr)
        """
        # Arrange
        token = 'test_token_xyz789'

        # Act
        service = GitLabService(
            token=token,
            external_token_manager=True,
        )

        # Assert
        # When external_token_manager is True, token should not be assigned
        # It should remain as empty string or SecretStr (the default)
        if isinstance(service.token, SecretStr):
            assert service.token.get_secret_value() == ''
        else:
            assert service.token == '' or service.token is None

    def test_external_manager_flag_is_stored(self):
        """
        Test: external_token_manager flag is stored correctly.

        Arrange: Create GitLabService with external_token_manager=True
        Act: Initialize the service
        Assert: external_token_manager attribute is True
        """
        # Arrange & Act
        service = GitLabService(
            token='test_token',
            external_token_manager=True,
        )

        # Assert
        assert service.external_token_manager is True
