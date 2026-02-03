"""Tests for OrgMemberService."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from server.services.org_member_service import OrgMemberService
from storage.org_member import OrgMember
from storage.role import Role


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
