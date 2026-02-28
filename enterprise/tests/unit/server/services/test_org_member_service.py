"""Tests for OrgMemberService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from server.routes.org_models import (
    CannotModifySelfError,
    InvalidRoleError,
    LastOwnerError,
    MeResponse,
    OrgMemberNotFoundError,
    OrgMemberResponse,
    OrgMemberUpdate,
    RoleNotFoundError,
)
from server.services.org_member_service import OrgMemberService
from storage.org_member import OrgMember
from storage.role import Role
from storage.user import User


@pytest.fixture
def org_id():
    """Create a test organization ID."""
    return uuid.uuid4()


@pytest.fixture
def current_user_id():
    """Create a test current user ID."""
    return uuid.uuid4()


@pytest.fixture
def target_user_id():
    """Create a test target user ID."""
    return uuid.uuid4()


@pytest.fixture
def owner_role():
    """Create a mock owner role."""
    role = MagicMock(spec=Role)
    role.id = 1
    role.name = 'owner'
    role.rank = 10
    return role


@pytest.fixture
def admin_role():
    """Create a mock admin role."""
    role = MagicMock(spec=Role)
    role.id = 2
    role.name = 'admin'
    role.rank = 20
    return role


@pytest.fixture
def member_role():
    """Create a mock member role."""
    role = MagicMock(spec=Role)
    role.id = 3
    role.name = 'member'
    role.rank = 1000
    return role


@pytest.fixture
def requester_membership_owner(org_id, current_user_id, owner_role):
    """Create a mock requester membership with owner role."""
    membership = MagicMock(spec=OrgMember)
    membership.org_id = org_id
    membership.user_id = current_user_id
    membership.role_id = owner_role.id
    return membership


@pytest.fixture
def requester_membership_admin(org_id, current_user_id, admin_role):
    """Create a mock requester membership with admin role."""
    membership = MagicMock(spec=OrgMember)
    membership.org_id = org_id
    membership.user_id = current_user_id
    membership.role_id = admin_role.id
    return membership


@pytest.fixture
def target_membership_user(org_id, target_user_id, member_role):
    """Create a mock target membership with user role."""
    membership = MagicMock(spec=OrgMember)
    membership.org_id = org_id
    membership.user_id = target_user_id
    membership.role_id = member_role.id
    return membership


@pytest.fixture
def target_membership_admin(org_id, target_user_id, admin_role):
    """Create a mock target membership with admin role."""
    membership = MagicMock(spec=OrgMember)
    membership.org_id = org_id
    membership.user_id = target_user_id
    membership.role_id = admin_role.id
    return membership


class TestOrgMemberServiceGetOrgMembers:
    """Test cases for OrgMemberService.get_org_members."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = MagicMock()
        user.email = 'test@example.com'
        return user

    @pytest.fixture
    def mock_role(self):
        """Create a mock role."""
        role = MagicMock(spec=Role)
        role.id = 1
        role.name = 'owner'
        role.rank = 10
        return role

    @pytest.fixture
    def mock_org_member(self, org_id, current_user_id, mock_user, mock_role):
        """Create a mock org member with user and role."""
        member = MagicMock(spec=OrgMember)
        member.org_id = org_id
        member.user_id = current_user_id
        member.role_id = mock_role.id
        member.status = 'active'
        member.user = mock_user
        member.role = mock_role
        return member

    @pytest.mark.asyncio
    async def test_get_members_succeeds_returns_paginated_data(
        self, org_id, current_user_id, mock_org_member, requester_membership_owner
    ):
        """Test that successful retrieval returns paginated member data."""
        # Arrange
        from server.routes.org_models import OrgMemberPage

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([mock_org_member], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is True
            assert error_code is None
            assert data is not None
            assert isinstance(data, OrgMemberPage)
            assert len(data.items) == 1
            assert data.current_page == 1
            assert data.per_page == 100
            assert data.items[0].user_id == str(current_user_id)
            assert data.items[0].email == 'test@example.com'
            assert data.items[0].role_id == 1
            assert data.items[0].role == 'owner'
            assert data.items[0].role_rank == 10
            assert data.items[0].status == 'active'

    @pytest.mark.asyncio
    async def test_user_not_a_member_returns_error(self, org_id, current_user_id):
        """Test that retrieval fails when user is not a member."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = None

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is False
            assert error_code == 'not_a_member'
            assert data is None

    @pytest.mark.asyncio
    async def test_invalid_page_id_negative_returns_error(
        self, org_id, current_user_id, requester_membership_owner
    ):
        """Test that negative page_id returns error."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = requester_membership_owner

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id='-1',
                limit=100,
            )

            # Assert
            assert success is False
            assert error_code == 'invalid_page_id'
            assert data is None

    @pytest.mark.asyncio
    async def test_invalid_page_id_non_integer_returns_error(
        self, org_id, current_user_id, requester_membership_owner
    ):
        """Test that non-integer page_id returns error."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = requester_membership_owner

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id='not-a-number',
                limit=100,
            )

            # Assert
            assert success is False
            assert error_code == 'invalid_page_id'
            assert data is None

    @pytest.mark.asyncio
    async def test_first_page_pagination_no_page_id(
        self, org_id, current_user_id, mock_org_member, requester_membership_owner
    ):
        """Test first page pagination when page_id is None."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([mock_org_member], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is True
            assert data is not None
            assert data.current_page == 1
            mock_get_paginated.assert_called_once_with(
                org_id=org_id, offset=0, limit=100, email_filter=None
            )

    @pytest.mark.asyncio
    async def test_next_page_pagination_with_page_id(
        self, org_id, current_user_id, mock_org_member, requester_membership_owner
    ):
        """Test next page pagination when page_id is provided."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([mock_org_member], True)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id='100',
                limit=50,
            )

            # Assert
            assert success is True
            assert data is not None
            assert data.current_page == 3  # offset (100) / limit (50) + 1
            mock_get_paginated.assert_called_once_with(
                org_id=org_id, offset=100, limit=50, email_filter=None
            )

    @pytest.mark.asyncio
    async def test_last_page_has_more_false(
        self, org_id, current_user_id, mock_org_member, requester_membership_owner
    ):
        """Test last page when has_more is False."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([mock_org_member], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id='200',
                limit=100,
            )

            # Assert
            assert success is True
            assert data is not None
            assert data.current_page == 3

    @pytest.mark.asyncio
    async def test_empty_organization_no_members(
        self, org_id, current_user_id, requester_membership_owner
    ):
        """Test empty organization with no members."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is True
            assert data is not None
            assert len(data.items) == 0

    @pytest.mark.asyncio
    async def test_missing_user_relationship_handles_gracefully(
        self, org_id, current_user_id, mock_role, requester_membership_owner
    ):
        """Test that missing user relationship is handled gracefully."""
        # Arrange
        member_no_user = MagicMock(spec=OrgMember)
        member_no_user.org_id = org_id
        member_no_user.user_id = current_user_id
        member_no_user.role_id = mock_role.id
        member_no_user.status = 'active'
        member_no_user.user = None
        member_no_user.role = mock_role

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([member_no_user], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is True
            assert data is not None
            assert len(data.items) == 1
            assert data.items[0].email is None

    @pytest.mark.asyncio
    async def test_missing_role_relationship_handles_gracefully(
        self, org_id, current_user_id, mock_user, requester_membership_owner
    ):
        """Test that missing role relationship is handled gracefully."""
        # Arrange
        member_no_role = MagicMock(spec=OrgMember)
        member_no_role.org_id = org_id
        member_no_role.user_id = current_user_id
        member_no_role.role_id = 1
        member_no_role.status = 'active'
        member_no_role.user = mock_user
        member_no_role.role = None

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([member_no_role], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is True
            assert data is not None
            assert len(data.items) == 1
            assert data.items[0].role == ''
            assert data.items[0].role_rank == 0

    @pytest.mark.asyncio
    async def test_multiple_members_returns_all(
        self, org_id, current_user_id, mock_user, mock_role, requester_membership_owner
    ):
        """Test that multiple members are returned correctly."""
        # Arrange
        member1 = MagicMock(spec=OrgMember)
        member1.org_id = org_id
        member1.user_id = current_user_id
        member1.role_id = mock_role.id
        member1.status = 'active'
        member1.user = mock_user
        member1.role = mock_role

        member2 = MagicMock(spec=OrgMember)
        member2.org_id = org_id
        member2.user_id = uuid.uuid4()
        member2.role_id = mock_role.id
        member2.status = 'active'
        member2.user = mock_user
        member2.role = mock_role

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([member1, member2], False)

            # Act
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=100,
            )

            # Assert
            assert success is True
            assert data is not None
            assert len(data.items) == 2

    @pytest.mark.asyncio
    async def test_email_filter_passed_to_store(
        self, org_id, current_user_id, mock_org_member, requester_membership_owner
    ):
        """Test that email filter is passed to store methods."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([mock_org_member], False)

            # Act
            await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id=None,
                limit=10,
                email_filter='alice',
            )

            # Assert
            mock_get_paginated.assert_called_once_with(
                org_id=org_id, offset=0, limit=10, email_filter='alice'
            )

    @pytest.mark.asyncio
    async def test_pagination_metadata_correct_for_page_2(
        self, org_id, current_user_id, mock_org_member, requester_membership_owner
    ):
        """Test pagination metadata is correct for page 2."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_paginated',
                new_callable=AsyncMock,
            ) as mock_get_paginated,
        ):
            mock_get_member.return_value = requester_membership_owner
            mock_get_paginated.return_value = ([mock_org_member], True)

            # Act - Request page 2 (offset 10) with limit 10
            success, error_code, data = await OrgMemberService.get_org_members(
                org_id=org_id,
                current_user_id=current_user_id,
                page_id='10',
                limit=10,
            )

            # Assert
            assert success is True
            assert data is not None
            assert data.current_page == 2
            assert data.per_page == 10


class TestOrgMemberServiceGetOrgMembersCount:
    """Test cases for OrgMemberService.get_org_members_count."""

    @pytest.fixture
    def requester_membership(self, org_id, current_user_id):
        """Create a mock requester membership."""
        membership = MagicMock(spec=OrgMember)
        membership.org_id = org_id
        membership.user_id = current_user_id
        membership.role_id = 1
        return membership

    @pytest.mark.asyncio
    async def test_count_succeeds_returns_count(
        self, org_id, current_user_id, requester_membership
    ):
        """Test that successful count returns the member count."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_count',
                new_callable=AsyncMock,
            ) as mock_get_count,
        ):
            mock_get_member.return_value = requester_membership
            mock_get_count.return_value = 42

            # Act
            count = await OrgMemberService.get_org_members_count(
                org_id=org_id,
                current_user_id=current_user_id,
            )

            # Assert
            assert count == 42
            mock_get_count.assert_called_once_with(org_id=org_id, email_filter=None)

    @pytest.mark.asyncio
    async def test_count_with_email_filter(
        self, org_id, current_user_id, requester_membership
    ):
        """Test that email filter is passed to store method."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members_count',
                new_callable=AsyncMock,
            ) as mock_get_count,
        ):
            mock_get_member.return_value = requester_membership
            mock_get_count.return_value = 5

            # Act
            count = await OrgMemberService.get_org_members_count(
                org_id=org_id,
                current_user_id=current_user_id,
                email_filter='alice',
            )

            # Assert
            assert count == 5
            mock_get_count.assert_called_once_with(org_id=org_id, email_filter='alice')

    @pytest.mark.asyncio
    async def test_not_a_member_raises_error(self, org_id, current_user_id):
        """Test that non-member raises OrgMemberNotFoundError."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = None

            # Act & Assert
            with pytest.raises(OrgMemberNotFoundError):
                await OrgMemberService.get_org_members_count(
                    org_id=org_id,
                    current_user_id=current_user_id,
                )


@pytest.fixture
def target_membership_owner(org_id, target_user_id, owner_role):
    """Create a mock target membership with owner role."""
    membership = MagicMock(spec=OrgMember)
    membership.org_id = org_id
    membership.user_id = target_user_id
    membership.role_id = owner_role.id
    return membership


class TestOrgMemberServiceRemoveOrgMember:
    """Test cases for OrgMemberService.remove_org_member."""

    @pytest.mark.asyncio
    async def test_owner_removes_user_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that an owner can successfully remove a regular user."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None
            mock_remove.assert_called_once_with(org_id, target_user_id)

    @pytest.mark.asyncio
    async def test_owner_removes_admin_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_admin,
        owner_role,
        admin_role,
    ):
        """Test that an owner can successfully remove an admin."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_admin,
            ]
            mock_get_role.side_effect = [owner_role, admin_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_admin_removes_user_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_user,
        admin_role,
        member_role,
    ):
        """Test that an admin can successfully remove a regular user."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_admin,
                target_membership_user,
            ]
            mock_get_role.side_effect = [admin_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_requester_not_a_member_returns_error(
        self, org_id, current_user_id, target_user_id
    ):
        """Test that removing fails when requester is not a member of the organization."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'not_a_member'

    @pytest.mark.asyncio
    async def test_cannot_remove_self_returns_error(
        self, org_id, current_user_id, requester_membership_owner, owner_role
    ):
        """Test that removing fails when trying to remove oneself."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = requester_membership_owner

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, current_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'cannot_remove_self'

    @pytest.mark.asyncio
    async def test_target_member_not_found_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        owner_role,
    ):
        """Test that removing fails when target member is not found."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.side_effect = [requester_membership_owner, None]

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'member_not_found'

    @pytest.mark.asyncio
    async def test_role_not_found_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
    ):
        """Test that removing fails when role is not found."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, None]

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'role_not_found'

    @pytest.mark.asyncio
    async def test_admin_can_remove_admin_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_admin,
        admin_role,
    ):
        """Test that an admin can remove another admin."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch(
                'server.services.org_member_service.LiteLlmManager.remove_user_from_team'
            ) as mock_remove_litellm,
        ):
            mock_get_member.side_effect = [
                requester_membership_admin,
                target_membership_admin,
            ]
            mock_get_role.side_effect = [admin_role, admin_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None
            mock_remove_litellm.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_admin_cannot_remove_owner_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_owner,
        admin_role,
        owner_role,
    ):
        """Test that an admin cannot remove an owner."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_member.side_effect = [
                requester_membership_admin,
                target_membership_owner,
            ]
            mock_get_role.side_effect = [admin_role, owner_role]

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'insufficient_permission'

    @pytest.mark.asyncio
    async def test_user_cannot_remove_anyone_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_user,
        member_role,
    ):
        """Test that a regular user cannot remove anyone."""
        # Arrange
        requester_membership_user = MagicMock(spec=OrgMember)
        requester_membership_user.org_id = org_id
        requester_membership_user.user_id = current_user_id
        requester_membership_user.role_id = member_role.id

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_member.side_effect = [
                requester_membership_user,
                target_membership_user,
            ]
            mock_get_role.side_effect = [member_role, member_role]

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'insufficient_permission'

    @pytest.mark.asyncio
    async def test_cannot_remove_last_owner_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_owner,
        owner_role,
    ):
        """Test that removing the last owner fails."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members'
            ) as mock_get_members,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_owner,
            ]
            mock_get_role.return_value = owner_role
            # Only one owner (the target)
            mock_get_members.return_value = [target_membership_owner]

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'cannot_remove_last_owner'

    @pytest.mark.asyncio
    async def test_can_remove_owner_when_multiple_owners_exist(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_owner,
        owner_role,
    ):
        """Test that an owner can be removed when there are multiple owners."""
        # Arrange
        another_owner = MagicMock(spec=OrgMember)
        another_owner.user_id = uuid.uuid4()
        another_owner.role_id = owner_role.id

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members'
            ) as mock_get_members,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_owner,
            ]
            mock_get_role.return_value = owner_role
            # Multiple owners exist
            mock_get_members.return_value = [
                requester_membership_owner,
                target_membership_owner,
                another_owner,
            ]
            mock_remove.return_value = True
            mock_get_user.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_removal_failed_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that removing fails when store removal returns False."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = False

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'removal_failed'

    @pytest.mark.asyncio
    async def test_remove_member_updates_current_org_id_when_matching(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that current_org_id is updated to personal workspace when it matches removed org."""
        # Arrange
        mock_user = MagicMock(spec=User)
        mock_user.current_org_id = (
            org_id  # User's current org matches the org being removed
        )

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch(
                'server.services.org_member_service.UserStore.update_current_org'
            ) as mock_update_org,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = mock_user

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None
            mock_update_org.assert_called_once_with(str(target_user_id), target_user_id)

    @pytest.mark.asyncio
    async def test_remove_member_does_not_update_current_org_id_when_not_matching(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that current_org_id is NOT updated when it differs from removed org."""
        # Arrange
        different_org_id = uuid.uuid4()
        mock_user = MagicMock(spec=User)
        mock_user.current_org_id = different_org_id  # User's current org is different

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch(
                'server.services.org_member_service.UserStore.update_current_org'
            ) as mock_update_org,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = mock_user

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None
            mock_update_org.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_member_succeeds_when_user_not_found_after_removal(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that removal succeeds even if user lookup returns None after removal."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch(
                'server.services.org_member_service.UserStore.update_current_org'
            ) as mock_update_org,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None  # User not found

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None
            mock_update_org.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_removal_calls_litellm_remove_user_from_team(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that LiteLLM remove_user_from_team is called after successful database removal."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch(
                'server.services.org_member_service.LiteLlmManager.remove_user_from_team',
                new_callable=AsyncMock,
            ) as mock_litellm_remove,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            mock_litellm_remove.assert_called_once_with(
                str(target_user_id), str(org_id)
            )

    @pytest.mark.asyncio
    async def test_litellm_failure_does_not_fail_removal(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that LiteLLM failure doesn't fail the overall removal operation."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch(
                'server.services.org_member_service.LiteLlmManager.remove_user_from_team',
                new_callable=AsyncMock,
            ) as mock_litellm_remove,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = True
            mock_get_user.return_value = None
            mock_litellm_remove.side_effect = Exception('LiteLLM API error')

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_database_failure_skips_litellm_call(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """Test that LiteLLM is not called when database removal fails."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.OrgMemberStore.remove_user_from_org'
            ) as mock_remove,
            patch(
                'server.services.org_member_service.LiteLlmManager.remove_user_from_team',
                new_callable=AsyncMock,
            ) as mock_litellm_remove,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_remove.return_value = False

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            mock_litellm_remove.assert_not_called()


class TestOrgMemberServiceCanRemoveMember:
    """Test cases for OrgMemberService._can_remove_member."""

    def test_owner_can_remove_admin(self):
        """Test that owner can remove admin."""
        # Act
        result = OrgMemberService._can_remove_member('owner', 'admin')

        # Assert
        assert result is True

    def test_owner_can_remove_user(self):
        """Test that owner can remove user."""
        # Act
        result = OrgMemberService._can_remove_member('owner', 'member')

        # Assert
        assert result is True

    def test_admin_can_remove_member(self):
        """Test that admin can remove member."""
        # Act
        result = OrgMemberService._can_remove_member('admin', 'member')

        # Assert
        assert result is True

    def test_admin_can_remove_admin(self):
        """Test that admin can remove another admin."""
        # Act
        result = OrgMemberService._can_remove_member('admin', 'admin')

        # Assert
        assert result is True

    def test_admin_cannot_remove_owner(self):
        """Test that admin cannot remove owner."""
        # Act
        result = OrgMemberService._can_remove_member('admin', 'owner')

        # Assert
        assert result is False

    def test_member_cannot_remove_anyone(self):
        """Test that member cannot remove anyone."""
        # Act
        result = OrgMemberService._can_remove_member('member', 'member')

        # Assert
        assert result is False


class TestOrgMemberServiceUpdateOrgMember:
    """Test cases for OrgMemberService.update_org_member."""

    @pytest.mark.asyncio
    async def test_owner_updates_user_to_admin_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
        admin_role,
    ):
        """GIVEN owner and target user WHEN owner sets target role to admin THEN update succeeds and returns OrgMemberResponse."""
        # Arrange
        updated_member = MagicMock(spec=OrgMember)
        updated_member.user_id = target_user_id
        updated_member.role_id = admin_role.id
        updated_member.status = 'active'
        mock_user = MagicMock()
        mock_user.email = 'target@example.com'
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_name'
            ) as mock_get_role_by_name,
            patch(
                'server.services.org_member_service.OrgMemberStore.update_user_role_in_org'
            ) as mock_update,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_get_role_by_name.return_value = admin_role
            mock_update.return_value = updated_member
            mock_get_user.return_value = mock_user

            # Act
            data = await OrgMemberService.update_org_member(
                org_id, target_user_id, current_user_id, OrgMemberUpdate(role='admin')
            )

            # Assert
            assert isinstance(data, OrgMemberResponse)
            assert data.role == 'admin'
            assert data.role_rank == 20
            mock_update.assert_called_once_with(org_id, target_user_id, admin_role.id)

    @pytest.mark.asyncio
    async def test_admin_updates_user_to_admin_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_user,
        admin_role,
        member_role,
    ):
        """GIVEN admin and target user WHEN admin sets target role to admin THEN update succeeds."""
        # Arrange
        updated_member = MagicMock(spec=OrgMember)
        updated_member.user_id = target_user_id
        updated_member.role_id = admin_role.id
        updated_member.status = 'active'
        mock_user = MagicMock()
        mock_user.email = 'target@example.com'
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_name'
            ) as mock_get_role_by_name,
            patch(
                'server.services.org_member_service.OrgMemberStore.update_user_role_in_org'
            ) as mock_update,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_admin,
                target_membership_user,
            ]
            mock_get_role.side_effect = [admin_role, member_role]
            mock_get_role_by_name.return_value = admin_role
            mock_update.return_value = updated_member
            mock_get_user.return_value = mock_user

            # Act
            data = await OrgMemberService.update_org_member(
                org_id, target_user_id, current_user_id, OrgMemberUpdate(role='admin')
            )

            # Assert
            assert data is not None
            mock_update.assert_called_once_with(org_id, target_user_id, admin_role.id)

    @pytest.mark.asyncio
    async def test_admin_can_update_admin_to_member_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_admin,
        admin_role,
        member_role,
    ):
        """GIVEN admin and target admin WHEN admin changes target role to member THEN update succeeds."""
        # Arrange
        updated_member = MagicMock(spec=OrgMember)
        updated_member.user_id = target_user_id
        updated_member.role_id = member_role.id
        updated_member.status = 'active'
        mock_user = MagicMock()
        mock_user.email = 'target@example.com'
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_name'
            ) as mock_get_role_by_name,
            patch(
                'server.services.org_member_service.OrgMemberStore.update_user_role_in_org'
            ) as mock_update,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_admin,
                target_membership_admin,
            ]
            mock_get_role.side_effect = [admin_role, admin_role]
            mock_get_role_by_name.return_value = member_role
            mock_update.return_value = updated_member
            mock_get_user.return_value = mock_user

            # Act
            data = await OrgMemberService.update_org_member(
                org_id,
                target_user_id,
                current_user_id,
                OrgMemberUpdate(role='member'),
            )

            # Assert
            assert isinstance(data, OrgMemberResponse)
            assert data.role == 'member'
            mock_update.assert_called_once_with(org_id, target_user_id, member_role.id)

    @pytest.mark.asyncio
    async def test_owner_can_update_owner_to_admin_succeeds(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_owner,
        owner_role,
        admin_role,
    ):
        """GIVEN owner and target owner WHEN owner changes target role to admin THEN update succeeds."""
        # Arrange
        updated_member = MagicMock(spec=OrgMember)
        updated_member.user_id = target_user_id
        updated_member.role_id = admin_role.id
        updated_member.status = 'active'
        mock_user = MagicMock()
        mock_user.email = 'target@example.com'
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_name'
            ) as mock_get_role_by_name,
            patch(
                'server.services.org_member_service.OrgMemberStore.update_user_role_in_org'
            ) as mock_update,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
            patch.object(OrgMemberService, '_is_last_owner', return_value=False),
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_owner,
            ]
            mock_get_role.side_effect = [owner_role, owner_role]
            mock_get_role_by_name.return_value = admin_role
            mock_update.return_value = updated_member
            mock_get_user.return_value = mock_user

            # Act
            data = await OrgMemberService.update_org_member(
                org_id,
                target_user_id,
                current_user_id,
                OrgMemberUpdate(role='admin'),
            )

            # Assert
            assert isinstance(data, OrgMemberResponse)
            assert data.role == 'admin'
            mock_update.assert_called_once_with(org_id, target_user_id, admin_role.id)

    @pytest.mark.asyncio
    async def test_requester_not_a_member_raises_error(
        self, org_id, current_user_id, target_user_id
    ):
        """GIVEN requester not in org WHEN update_org_member THEN raises OrgMemberNotFoundError."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = None

            # Act & Assert
            with pytest.raises(OrgMemberNotFoundError):
                await OrgMemberService.update_org_member(
                    org_id,
                    target_user_id,
                    current_user_id,
                    OrgMemberUpdate(role='member'),
                )

    @pytest.mark.asyncio
    async def test_cannot_modify_self_raises_error(
        self, org_id, current_user_id, requester_membership_owner, owner_role
    ):
        """GIVEN requester updates self WHEN update_org_member THEN raises CannotModifySelfError."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = requester_membership_owner

            # Act & Assert
            with pytest.raises(CannotModifySelfError):
                await OrgMemberService.update_org_member(
                    org_id,
                    current_user_id,
                    current_user_id,
                    OrgMemberUpdate(role='member'),
                )

    @pytest.mark.asyncio
    async def test_target_member_not_found_raises_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        owner_role,
    ):
        """GIVEN target not in org WHEN update_org_member THEN raises OrgMemberNotFoundError."""
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.side_effect = [requester_membership_owner, None]

            # Act & Assert
            with pytest.raises(OrgMemberNotFoundError):
                await OrgMemberService.update_org_member(
                    org_id,
                    target_user_id,
                    current_user_id,
                    OrgMemberUpdate(role='member'),
                )

    @pytest.mark.asyncio
    async def test_invalid_role_name_raises_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """GIVEN unknown role name WHEN update_org_member THEN raises InvalidRoleError."""
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_name'
            ) as mock_get_role_by_name,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_get_role_by_name.return_value = None

            # Act & Assert
            with pytest.raises(InvalidRoleError):
                await OrgMemberService.update_org_member(
                    org_id,
                    target_user_id,
                    current_user_id,
                    OrgMemberUpdate(role='superuser'),
                )

    @pytest.mark.asyncio
    async def test_cannot_demote_last_owner_raises_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_owner,
        owner_role,
        admin_role,
    ):
        """GIVEN last owner would be demoted WHEN update_org_member THEN raises LastOwnerError."""
        # Arrange: patch _can_update_member_role so we reach the last-owner check (owner cannot normally modify owner)
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_name'
            ) as mock_get_role_by_name,
            patch(
                'server.services.org_member_service.OrgMemberService._can_update_member_role'
            ) as mock_can_update,
            patch(
                'server.services.org_member_service.OrgMemberService._is_last_owner'
            ) as mock_is_last_owner,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_owner,
            ]
            mock_get_role.side_effect = [owner_role, owner_role]
            mock_get_role_by_name.return_value = admin_role
            mock_can_update.return_value = True
            mock_is_last_owner.return_value = True

            # Act & Assert
            with pytest.raises(LastOwnerError):
                await OrgMemberService.update_org_member(
                    org_id,
                    target_user_id,
                    current_user_id,
                    OrgMemberUpdate(role='admin'),
                )

    @pytest.mark.asyncio
    async def test_no_role_in_body_returns_current_member_state(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_owner,
        target_membership_user,
        owner_role,
        member_role,
    ):
        """GIVEN update with no role WHEN update_org_member THEN returns current member without changing role."""
        # Arrange
        mock_user = MagicMock()
        mock_user.email = 'target@example.com'
        target_membership_user.status = 'active'
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, member_role]
            mock_get_user.return_value = mock_user

            # Act
            data = await OrgMemberService.update_org_member(
                org_id, target_user_id, current_user_id, OrgMemberUpdate(role=None)
            )

            # Assert
            assert data is not None
            assert data.role == 'member'
            assert data.role_rank == 1000


class TestOrgMemberServiceCanUpdateMemberRole:
    """Test cases for OrgMemberService._can_update_member_role."""

    def test_owner_can_set_any_role_for_non_owner(self):
        """Owner can change admin/user target to any role."""
        assert (
            OrgMemberService._can_update_member_role('owner', 'admin', 'owner') is True
        )
        assert (
            OrgMemberService._can_update_member_role('owner', 'admin', 'admin') is True
        )
        assert (
            OrgMemberService._can_update_member_role('owner', 'member', 'owner') is True
        )

    def test_owner_can_modify_owner(self):
        """Owner can change another owner's role."""
        assert (
            OrgMemberService._can_update_member_role('owner', 'owner', 'admin') is True
        )

    def test_admin_can_set_admin_or_member_for_member(self):
        """Admin can set admin or member role for a member target."""
        assert (
            OrgMemberService._can_update_member_role('admin', 'member', 'admin') is True
        )
        assert (
            OrgMemberService._can_update_member_role('admin', 'member', 'member')
            is True
        )

    def test_admin_can_modify_admin(self):
        """Admin can modify another admin's role to member."""
        assert (
            OrgMemberService._can_update_member_role('admin', 'admin', 'member') is True
        )

    def test_admin_cannot_modify_owner(self):
        """Admin cannot modify owner targets."""
        assert (
            OrgMemberService._can_update_member_role('admin', 'owner', 'admin') is False
        )

    def test_admin_cannot_set_owner_role(self):
        """Admin cannot set role to owner."""
        assert (
            OrgMemberService._can_update_member_role('admin', 'member', 'owner')
            is False
        )

    def test_member_cannot_update_anyone(self):
        """Member cannot update any member's role."""
        assert (
            OrgMemberService._can_update_member_role('member', 'member', 'admin')
            is False
        )


class TestOrgMemberServiceIsLastOwner:
    """Test cases for OrgMemberService._is_last_owner."""

    def test_is_last_owner_when_only_one_owner(
        self, org_id, target_user_id, owner_role
    ):
        """Test that returns True when user is the only owner."""
        # Arrange
        target_membership = MagicMock(spec=OrgMember)
        target_membership.user_id = target_user_id
        target_membership.role_id = owner_role.id

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members'
            ) as mock_get_members,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_members.return_value = [target_membership]
            mock_get_role.return_value = owner_role

            # Act
            result = OrgMemberService._is_last_owner(org_id, target_user_id)

            # Assert
            assert result is True

    def test_is_not_last_owner_when_multiple_owners(
        self, org_id, target_user_id, owner_role
    ):
        """Test that returns False when there are multiple owners."""
        # Arrange
        target_membership = MagicMock(spec=OrgMember)
        target_membership.user_id = target_user_id
        target_membership.role_id = owner_role.id

        another_owner = MagicMock(spec=OrgMember)
        another_owner.user_id = uuid.uuid4()
        another_owner.role_id = owner_role.id

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members'
            ) as mock_get_members,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_members.return_value = [target_membership, another_owner]
            mock_get_role.return_value = owner_role

            # Act
            result = OrgMemberService._is_last_owner(org_id, target_user_id)

            # Assert
            assert result is False

    def test_is_not_last_owner_when_user_is_not_owner(
        self, org_id, target_user_id, member_role
    ):
        """Test that returns False when user is not an owner."""
        # Arrange
        target_membership = MagicMock(spec=OrgMember)
        target_membership.user_id = target_user_id
        target_membership.role_id = member_role.id

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members'
            ) as mock_get_members,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_members.return_value = [target_membership]
            mock_get_role.return_value = member_role

            # Act
            result = OrgMemberService._is_last_owner(org_id, target_user_id)

            # Assert
            assert result is False


class TestOrgMemberServiceGetMe:
    """Test cases for OrgMemberService.get_me."""

    @pytest.fixture
    def mock_org_member(self, org_id, current_user_id):
        """Create a mock OrgMember with LLM fields."""
        member = MagicMock(spec=OrgMember)
        member.org_id = org_id
        member.user_id = current_user_id
        member.role_id = 1
        member.llm_api_key = SecretStr('sk-test-key-12345')
        member.llm_api_key_for_byor = None
        member.llm_model = 'gpt-4'
        member.llm_base_url = 'https://api.example.com'
        member.max_iterations = 50
        member.status = 'active'
        return member

    @pytest.fixture
    def mock_user(self, current_user_id):
        """Create a mock User."""
        user = MagicMock(spec=User)
        user.id = current_user_id
        user.email = 'test@example.com'
        return user

    def test_get_me_success_returns_me_response(
        self, org_id, current_user_id, mock_org_member, mock_user, owner_role
    ):
        """GIVEN: User is a member of the organization
        WHEN: get_me is called
        THEN: Returns MeResponse with user's membership data
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.return_value = mock_org_member
            mock_get_role.return_value = owner_role
            mock_get_user.return_value = mock_user

            # Act
            result = OrgMemberService.get_me(org_id, current_user_id)

            # Assert
            assert isinstance(result, MeResponse)
            assert result.org_id == str(org_id)
            assert result.user_id == str(current_user_id)
            assert result.email == 'test@example.com'
            assert result.role == 'owner'
            assert result.llm_model == 'gpt-4'
            assert result.max_iterations == 50
            assert result.status == 'active'

    def test_get_me_member_not_found_raises_error(self, org_id, current_user_id):
        """GIVEN: User is not a member of the organization
        WHEN: get_me is called
        THEN: Raises OrgMemberNotFoundError
        """
        # Arrange
        with patch(
            'server.services.org_member_service.OrgMemberStore.get_org_member'
        ) as mock_get_member:
            mock_get_member.return_value = None

            # Act & Assert
            with pytest.raises(OrgMemberNotFoundError) as exc_info:
                OrgMemberService.get_me(org_id, current_user_id)

            assert str(org_id) in str(exc_info.value)

    def test_get_me_role_not_found_raises_error(
        self, org_id, current_user_id, mock_org_member
    ):
        """GIVEN: Member exists but role lookup fails
        WHEN: get_me is called
        THEN: Raises RoleNotFoundError
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_member.return_value = mock_org_member
            mock_get_role.return_value = None

            # Act & Assert
            with pytest.raises(RoleNotFoundError) as exc_info:
                OrgMemberService.get_me(org_id, current_user_id)

            assert exc_info.value.role_id == mock_org_member.role_id

    def test_get_me_user_not_found_returns_empty_email(
        self, org_id, current_user_id, mock_org_member, owner_role
    ):
        """GIVEN: Member exists but user lookup returns None
        WHEN: get_me is called
        THEN: Returns MeResponse with empty email
        """
        # Arrange
        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
            patch(
                'server.services.org_member_service.UserStore.get_user_by_id'
            ) as mock_get_user,
        ):
            mock_get_member.return_value = mock_org_member
            mock_get_role.return_value = owner_role
            mock_get_user.return_value = None

            # Act
            result = OrgMemberService.get_me(org_id, current_user_id)

            # Assert
            assert result.email == ''
