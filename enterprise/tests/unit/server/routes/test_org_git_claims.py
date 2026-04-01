"""Tests for Git organization claim API endpoints.

Tests the following endpoints:
- GET /api/organizations/{org_id}/git-claims (list claims)
- POST /api/organizations/{org_id}/git-claims (claim)
- DELETE /api/organizations/{org_id}/git-claims/{claim_id} (disconnect)
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from server.routes.orgs import (
    claim_git_organization,
    disconnect_git_organization,
    get_git_claims,
)
from storage.org_git_claim import OrgGitClaim


@pytest.fixture
def org_id():
    return uuid.uuid4()


@pytest.fixture
def user_id():
    return str(uuid.uuid4())


@pytest.fixture
def make_claim():
    """Factory to create mock OrgGitClaim objects."""

    def _make(org_id, provider='github', git_organization='OpenHands', claimed_by=None):
        claim = MagicMock(spec=OrgGitClaim)
        claim.id = uuid.uuid4()
        claim.org_id = org_id
        claim.provider = provider
        claim.git_organization = git_organization
        claim.claimed_by = claimed_by or uuid.uuid4()
        claim.claimed_at = datetime(2026, 4, 1, 12, 0, 0)
        return claim

    return _make


# =============================================================================
# GET /api/organizations/{org_id}/git-claims
# =============================================================================


class TestGetGitClaims:
    """Tests for the get Git organization claims endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_claims(self, org_id, user_id):
        """
        GIVEN: An organization with no Git claims
        WHEN: GET /api/organizations/{org_id}/git-claims is called
        THEN: An empty list is returned
        """
        with patch(
            'server.routes.orgs.OrgGitClaimStore.get_claims_by_org_id',
            AsyncMock(return_value=[]),
        ) as mock_get:
            result = await get_git_claims(org_id=org_id, user_id=user_id)

        assert result == []
        mock_get.assert_called_once_with(org_id=org_id)

    @pytest.mark.asyncio
    async def test_returns_claims_for_organization(self, org_id, user_id, make_claim):
        """
        GIVEN: An organization with multiple Git claims
        WHEN: GET /api/organizations/{org_id}/git-claims is called
        THEN: All claims are returned with correct details
        """
        claim1 = make_claim(org_id, provider='github', git_organization='OpenHands')
        claim2 = make_claim(org_id, provider='gitlab', git_organization='AcmeCo')

        with patch(
            'server.routes.orgs.OrgGitClaimStore.get_claims_by_org_id',
            AsyncMock(return_value=[claim1, claim2]),
        ):
            result = await get_git_claims(org_id=org_id, user_id=user_id)

        assert len(result) == 2
        assert result[0].id == str(claim1.id)
        assert result[0].org_id == str(org_id)
        assert result[0].provider == 'github'
        assert result[0].git_organization == 'OpenHands'
        assert result[0].claimed_by == str(claim1.claimed_by)
        assert result[0].claimed_at == '2026-04-01T12:00:00'
        assert result[1].id == str(claim2.id)
        assert result[1].provider == 'gitlab'
        assert result[1].git_organization == 'AcmeCo'

    @pytest.mark.asyncio
    async def test_returns_500_on_unexpected_error(self, org_id, user_id):
        """
        GIVEN: An unexpected error occurs when fetching claims
        WHEN: GET /api/organizations/{org_id}/git-claims is called
        THEN: A 500 Internal Server Error is returned
        """
        with patch(
            'server.routes.orgs.OrgGitClaimStore.get_claims_by_org_id',
            AsyncMock(side_effect=RuntimeError('db connection failed')),
        ):
            with pytest.raises(Exception) as exc_info:
                await get_git_claims(org_id=org_id, user_id=user_id)

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# POST /api/organizations/{org_id}/git-claims
# =============================================================================


class TestClaimGitOrganization:
    """Tests for the claim Git organization endpoint."""

    @pytest.mark.asyncio
    async def test_claim_succeeds_for_unclaimed_org(self, org_id, user_id, make_claim):
        """
        GIVEN: A Git organization that has not been claimed
        WHEN: POST /api/organizations/{org_id}/git-claims is called
        THEN: The claim is created and returned with correct details
        """
        # Arrange
        mock_claim = make_claim(org_id, claimed_by=uuid.UUID(user_id))
        request = MagicMock()
        request.provider = 'github'
        request.git_organization = 'OpenHands'

        with (
            patch(
                'server.routes.orgs.OrgGitClaimStore.get_claim_by_provider_and_git_org',
                AsyncMock(return_value=None),
            ),
            patch(
                'server.routes.orgs.OrgGitClaimStore.create_claim',
                AsyncMock(return_value=mock_claim),
            ) as mock_create,
        ):
            # Act
            response = await claim_git_organization(
                org_id=org_id, request=request, user_id=user_id
            )

        # Assert
        assert response.id == str(mock_claim.id)
        assert response.org_id == str(org_id)
        assert response.provider == 'github'
        assert response.git_organization == 'OpenHands'
        assert response.claimed_by == user_id
        mock_create.assert_called_once_with(
            org_id=org_id,
            provider='github',
            git_organization='OpenHands',
            claimed_by=uuid.UUID(user_id),
        )

    @pytest.mark.asyncio
    async def test_claim_fails_when_already_claimed(self, org_id, user_id, make_claim):
        """
        GIVEN: A Git organization already claimed by another OpenHands org
        WHEN: POST /api/organizations/{org_id}/git-claims is called
        THEN: A 409 Conflict error is returned
        """
        # Arrange
        other_org_id = uuid.uuid4()
        existing_claim = make_claim(
            other_org_id, provider='github', git_organization='AlreadyClaimed'
        )
        request = MagicMock()
        request.provider = 'github'
        request.git_organization = 'AlreadyClaimed'

        with patch(
            'server.routes.orgs.OrgGitClaimStore.get_claim_by_provider_and_git_org',
            AsyncMock(return_value=existing_claim),
        ):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await claim_git_organization(
                    org_id=org_id, request=request, user_id=user_id
                )

            assert exc_info.value.status_code == status.HTTP_409_CONFLICT

    @pytest.mark.asyncio
    async def test_claim_returns_500_on_unexpected_error(self, org_id, user_id):
        """
        GIVEN: An unexpected error occurs during claim creation
        WHEN: POST /api/organizations/{org_id}/git-claims is called
        THEN: A 500 Internal Server Error is returned
        """
        # Arrange
        request = MagicMock()
        request.provider = 'github'
        request.git_organization = 'OpenHands'

        with patch(
            'server.routes.orgs.OrgGitClaimStore.get_claim_by_provider_and_git_org',
            AsyncMock(side_effect=RuntimeError('db connection failed')),
        ):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await claim_git_organization(
                    org_id=org_id, request=request, user_id=user_id
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# DELETE /api/organizations/{org_id}/git-claims/{claim_id}
# =============================================================================


class TestDisconnectGitOrganization:
    """Tests for the disconnect Git organization endpoint."""

    @pytest.mark.asyncio
    async def test_disconnect_succeeds_for_existing_claim(self, org_id, user_id):
        """
        GIVEN: A valid claim belonging to the organization
        WHEN: DELETE /api/organizations/{org_id}/git-claims/{claim_id} is called
        THEN: The claim is deleted and a success message is returned
        """
        # Arrange
        claim_id = uuid.uuid4()

        with patch(
            'server.routes.orgs.OrgGitClaimStore.delete_claim',
            AsyncMock(return_value=True),
        ) as mock_delete:
            # Act
            result = await disconnect_git_organization(
                org_id=org_id, claim_id=claim_id, user_id=user_id
            )

        # Assert
        assert result == {'message': 'Git organization claim removed successfully'}
        mock_delete.assert_called_once_with(claim_id=claim_id, org_id=org_id)

    @pytest.mark.asyncio
    async def test_disconnect_fails_when_claim_not_found(self, org_id, user_id):
        """
        GIVEN: A claim_id that does not exist for this organization
        WHEN: DELETE /api/organizations/{org_id}/git-claims/{claim_id} is called
        THEN: A 404 Not Found error is returned
        """
        # Arrange
        claim_id = uuid.uuid4()

        with patch(
            'server.routes.orgs.OrgGitClaimStore.delete_claim',
            AsyncMock(return_value=False),
        ):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await disconnect_git_organization(
                    org_id=org_id, claim_id=claim_id, user_id=user_id
                )

            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_disconnect_returns_500_on_unexpected_error(self, org_id, user_id):
        """
        GIVEN: An unexpected error occurs during claim deletion
        WHEN: DELETE /api/organizations/{org_id}/git-claims/{claim_id} is called
        THEN: A 500 Internal Server Error is returned
        """
        # Arrange
        claim_id = uuid.uuid4()

        with patch(
            'server.routes.orgs.OrgGitClaimStore.delete_claim',
            AsyncMock(side_effect=RuntimeError('db connection failed')),
        ):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                await disconnect_git_organization(
                    org_id=org_id, claim_id=claim_id, user_id=user_id
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# Validation tests for GitOrgClaimRequest
# =============================================================================


class TestGitOrgClaimRequestValidation:
    """Tests for request model validation."""

    def test_valid_providers_are_accepted(self):
        """Each supported provider is accepted and normalized to lowercase."""
        from server.routes.org_models import GitOrgClaimRequest

        for provider in ['github', 'GitLab', 'BITBUCKET']:
            req = GitOrgClaimRequest(provider=provider, git_organization='test-org')
            assert req.provider == provider.lower().strip()

    def test_invalid_provider_is_rejected(self):
        """An unsupported provider raises a validation error."""
        from pydantic import ValidationError
        from server.routes.org_models import GitOrgClaimRequest

        with pytest.raises(ValidationError, match='Invalid provider'):
            GitOrgClaimRequest(provider='azure_devops', git_organization='test-org')

    def test_empty_git_organization_is_rejected(self):
        """An empty git_organization raises a validation error."""
        from pydantic import ValidationError
        from server.routes.org_models import GitOrgClaimRequest

        with pytest.raises(ValidationError, match='git_organization must not be empty'):
            GitOrgClaimRequest(provider='github', git_organization='   ')
