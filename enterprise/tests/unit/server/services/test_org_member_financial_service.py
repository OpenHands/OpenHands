"""Tests for OrgMemberFinancialService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.routes.org_models import OrgMemberFinancialPage
from server.services.org_member_financial_service import OrgMemberFinancialService
from storage.org_member import OrgMember


@pytest.fixture
def org_id():
    """Create a test organization ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock()
    user.email = 'test@example.com'
    return user


@pytest.fixture
def mock_role():
    """Create a mock role."""
    role = MagicMock()
    role.id = 1
    role.name = 'member'
    role.rank = 1000
    return role


@pytest.fixture
def mock_org_member(org_id, mock_user, mock_role):
    """Create a mock org member with user and role."""
    member = MagicMock(spec=OrgMember)
    member.org_id = org_id
    member.user_id = uuid.uuid4()
    member.role_id = mock_role.id
    member.status = 'active'
    member.user = mock_user
    member.role = mock_role
    return member


class TestOrgMemberFinancialServiceGetFinancialData:
    """Test cases for OrgMemberFinancialService.get_org_members_financial_data."""

    @pytest.mark.asyncio
    async def test_returns_paginated_financial_data_with_individual_budget(
        self, org_id, mock_org_member
    ):
        """
        GIVEN: Organization with members having individual budget limits
        WHEN: get_org_members_financial_data is called
        THEN: Returns financial data using individual spend for current_budget calc
        """
        # Arrange
        user_id_str = str(mock_org_member.user_id)
        litellm_data = {
            'team_max_budget': 1000.0,
            'team_spend': 200.0,
            'members': {
                user_id_str: {'spend': 125.50, 'max_budget': 500.0}  # Individual budget
            },
        }

        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 1)
            mock_get_financial.return_value = litellm_data

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
                page_id=None,
                limit=10,
            )

            # Assert
            assert isinstance(result, OrgMemberFinancialPage)
            assert len(result.items) == 1
            assert result.items[0].user_id == user_id_str
            assert result.items[0].email == 'test@example.com'
            assert result.items[0].lifetime_spend == 125.50
            assert result.items[0].max_budget == 500.0
            # Individual budget: 500 - 125.50 = 374.50
            assert result.items[0].current_budget == 374.50
            assert result.current_page == 1
            assert result.per_page == 10

    @pytest.mark.asyncio
    async def test_returns_shared_budget_using_team_spend(
        self, org_id, mock_org_member
    ):
        """
        GIVEN: Organization with shared team budget
        WHEN: get_org_members_financial_data is called
        THEN: Uses team_spend (not individual spend) for current_budget calculation
        """
        # Arrange
        user_id_str = str(mock_org_member.user_id)
        litellm_data = {
            'team_max_budget': 500.0,
            'team_spend': 150.0,  # Total team spend
            'members': {
                user_id_str: {
                    'spend': 50.0,
                    'max_budget': 500.0,
                    'uses_shared_budget': True,  # Explicitly using shared budget
                }
            },
        }

        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 1)
            mock_get_financial.return_value = litellm_data

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
            )

            # Assert
            assert len(result.items) == 1
            assert result.items[0].lifetime_spend == 50.0  # Individual spend
            assert result.items[0].max_budget == 500.0
            # Shared budget: 500 - 150 (team_spend) = 350
            assert result.items[0].current_budget == 350.0

    @pytest.mark.asyncio
    async def test_returns_defaults_when_litellm_data_missing(
        self, org_id, mock_org_member
    ):
        """
        GIVEN: Organization with members but no LiteLLM data for them
        WHEN: get_org_members_financial_data is called
        THEN: Returns financial data with default values (spend=0, budget=None)
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 1)
            mock_get_financial.return_value = {
                'team_max_budget': None,
                'team_spend': 0,
                'members': {},
            }

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
            )

            # Assert
            assert len(result.items) == 1
            assert result.items[0].lifetime_spend == 0
            assert result.items[0].max_budget is None
            assert result.items[0].current_budget == 0

    @pytest.mark.asyncio
    async def test_handles_litellm_failure_gracefully(self, org_id, mock_org_member):
        """
        GIVEN: LiteLLM service throws an exception
        WHEN: get_org_members_financial_data is called
        THEN: Returns financial data with default values (doesn't fail)
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 1)
            mock_get_financial.side_effect = Exception('LiteLLM unavailable')

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
            )

            # Assert - should not raise, returns defaults
            assert len(result.items) == 1
            assert result.items[0].lifetime_spend == 0
            assert result.items[0].max_budget is None

    @pytest.mark.asyncio
    async def test_pagination_returns_next_page_id(self, org_id, mock_org_member):
        """
        GIVEN: Organization with more members than limit
        WHEN: get_org_members_financial_data is called
        THEN: Returns next_page_id for pagination
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 25)  # 25 total
            mock_get_financial.return_value = {
                'team_max_budget': None,
                'team_spend': 0,
                'members': {},
            }

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
                page_id='0',
                limit=10,
            )

            # Assert
            assert result.current_page == 1
            assert result.next_page_id == '10'

    @pytest.mark.asyncio
    async def test_pagination_no_next_page_on_last_page(self, org_id, mock_org_member):
        """
        GIVEN: Organization on last page of results
        WHEN: get_org_members_financial_data is called
        THEN: Returns next_page_id as None
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 5)  # 5 total
            mock_get_financial.return_value = {
                'team_max_budget': None,
                'team_spend': 0,
                'members': {},
            }

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
                page_id='0',
                limit=10,
            )

            # Assert
            assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_empty_organization_returns_empty_items(self, org_id):
        """
        GIVEN: Organization with no members
        WHEN: get_org_members_financial_data is called
        THEN: Returns empty items list
        """
        # Arrange
        with patch(
            'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
            new_callable=AsyncMock,
        ) as mock_get_paginated:
            mock_get_paginated.return_value = ([], 0)

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
            )

            # Assert
            assert len(result.items) == 0
            assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_invalid_page_id_raises_value_error(self, org_id):
        """
        GIVEN: Invalid page_id format
        WHEN: get_org_members_financial_data is called
        THEN: Raises ValueError
        """
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
                page_id='invalid',
            )

        assert 'Invalid page_id' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_negative_page_id_raises_value_error(self, org_id):
        """
        GIVEN: Negative page_id
        WHEN: get_org_members_financial_data is called
        THEN: Raises ValueError
        """
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
                page_id='-5',
            )

        assert 'Invalid page_id' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_passes_email_filter_to_store(self, org_id, mock_org_member):
        """
        GIVEN: Email filter parameter
        WHEN: get_org_members_financial_data is called
        THEN: Passes email filter to the store
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([mock_org_member], 1)
            mock_get_financial.return_value = {
                'team_max_budget': None,
                'team_spend': 0,
                'members': {},
            }

            # Act
            await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
                email_filter='alice',
            )

            # Assert
            mock_get_paginated.assert_called_once_with(
                org_id=org_id, offset=0, limit=10, email_filter='alice'
            )

    @pytest.mark.asyncio
    async def test_handles_missing_user_relationship(self, org_id, mock_role):
        """
        GIVEN: Member with no user relationship loaded
        WHEN: get_org_members_financial_data is called
        THEN: Returns None for email
        """
        # Arrange
        member_no_user = MagicMock(spec=OrgMember)
        member_no_user.org_id = org_id
        member_no_user.user_id = uuid.uuid4()
        member_no_user.role_id = mock_role.id
        member_no_user.user = None  # No user relationship

        with (
            patch(
                'server.services.org_member_financial_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
            patch(
                'server.services.org_member_financial_service.LiteLlmManager.get_team_members_financial_data',
                new_callable=AsyncMock,
            ) as mock_get_financial,
        ):
            mock_get_paginated.return_value = ([member_no_user], 1)
            mock_get_financial.return_value = {
                'team_max_budget': None,
                'team_spend': 0,
                'members': {},
            }

            # Act
            result = await OrgMemberFinancialService.get_org_members_financial_data(
                org_id=org_id,
            )

            # Assert
            assert len(result.items) == 1
            assert result.items[0].email is None
