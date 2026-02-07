"""Tests for OrgMemberService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from server.routes.org_models import (
    MeResponse,
    OrgMemberNotFoundError,
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
def user_role():
    """Create a mock user role."""
    role = MagicMock(spec=Role)
    role.id = 3
    role.name = 'user'
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
def target_membership_user(org_id, target_user_id, user_role):
    """Create a mock target membership with user role."""
    membership = MagicMock(spec=OrgMember)
    membership.org_id = org_id
    membership.user_id = target_user_id
    membership.role_id = user_role.id
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
            assert data.next_page_id is None
            assert data.items[0].user_id == str(current_user_id)
            assert data.items[0].email == 'test@example.com'
            assert data.items[0].role_id == 1
            assert data.items[0].role_name == 'owner'
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
            assert data.next_page_id is None
            mock_get_paginated.assert_called_once_with(
                org_id=org_id, offset=0, limit=100
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
            assert data.next_page_id == '150'  # offset (100) + limit (50)
            mock_get_paginated.assert_called_once_with(
                org_id=org_id, offset=100, limit=50
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
            assert data.next_page_id is None

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
            assert data.next_page_id is None

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
            assert data.items[0].role_name == ''
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
        user_role,
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
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_user,
            ]
            mock_get_role.side_effect = [owner_role, user_role]
            mock_remove.return_value = True

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
        ):
            mock_get_member.side_effect = [
                requester_membership_owner,
                target_membership_admin,
            ]
            mock_get_role.side_effect = [owner_role, admin_role]
            mock_remove.return_value = True

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
        user_role,
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
        ):
            mock_get_member.side_effect = [
                requester_membership_admin,
                target_membership_user,
            ]
            mock_get_role.side_effect = [admin_role, user_role]
            mock_remove.return_value = True

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
    async def test_admin_cannot_remove_admin_returns_error(
        self,
        org_id,
        current_user_id,
        target_user_id,
        requester_membership_admin,
        target_membership_admin,
        admin_role,
    ):
        """Test that an admin cannot remove another admin."""
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
                target_membership_admin,
            ]
            mock_get_role.side_effect = [admin_role, admin_role]

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'insufficient_permission'

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
        user_role,
    ):
        """Test that a regular user cannot remove anyone."""
        # Arrange
        requester_membership_user = MagicMock(spec=OrgMember)
        requester_membership_user.org_id = org_id
        requester_membership_user.user_id = current_user_id
        requester_membership_user.role_id = user_role.id

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
            mock_get_role.side_effect = [user_role, user_role]

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
        user_role,
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
            mock_get_role.side_effect = [owner_role, user_role]
            mock_remove.return_value = False

            # Act
            success, error = await OrgMemberService.remove_org_member(
                org_id, target_user_id, current_user_id
            )

            # Assert
            assert success is False
            assert error == 'removal_failed'


class TestOrgMemberServiceCanRemoveMember:
    """Test cases for OrgMemberService._can_remove_member."""

    def test_owner_can_remove_admin(self):
        """Test that owner (rank 10) can remove admin (rank 20)."""
        # Arrange
        requester_rank = 10
        target_rank = 20

        # Act
        result = OrgMemberService._can_remove_member(requester_rank, target_rank)

        # Assert
        assert result is True

    def test_owner_can_remove_user(self):
        """Test that owner (rank 10) can remove user (rank 1000)."""
        # Arrange
        requester_rank = 10
        target_rank = 1000

        # Act
        result = OrgMemberService._can_remove_member(requester_rank, target_rank)

        # Assert
        assert result is True

    def test_admin_can_remove_user(self):
        """Test that admin (rank 20) can remove user (rank 1000)."""
        # Arrange
        requester_rank = 20
        target_rank = 1000

        # Act
        result = OrgMemberService._can_remove_member(requester_rank, target_rank)

        # Assert
        assert result is True

    def test_admin_cannot_remove_admin(self):
        """Test that admin (rank 20) cannot remove another admin (rank 20)."""
        # Arrange
        requester_rank = 20
        target_rank = 20

        # Act
        result = OrgMemberService._can_remove_member(requester_rank, target_rank)

        # Assert
        assert result is False

    def test_admin_cannot_remove_owner(self):
        """Test that admin (rank 20) cannot remove owner (rank 10)."""
        # Arrange
        requester_rank = 20
        target_rank = 10

        # Act
        result = OrgMemberService._can_remove_member(requester_rank, target_rank)

        # Assert
        assert result is False

    def test_user_cannot_remove_anyone(self):
        """Test that user (rank 1000) cannot remove anyone."""
        # Arrange
        requester_rank = 1000
        target_rank = 1000

        # Act
        result = OrgMemberService._can_remove_member(requester_rank, target_rank)

        # Assert
        assert result is False


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
        self, org_id, target_user_id, user_role
    ):
        """Test that returns False when user is not an owner."""
        # Arrange
        target_membership = MagicMock(spec=OrgMember)
        target_membership.user_id = target_user_id
        target_membership.role_id = user_role.id

        with (
            patch(
                'server.services.org_member_service.OrgMemberStore.get_org_members'
            ) as mock_get_members,
            patch(
                'server.services.org_member_service.RoleStore.get_role_by_id'
            ) as mock_get_role,
        ):
            mock_get_members.return_value = [target_membership]
            mock_get_role.return_value = user_role

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
