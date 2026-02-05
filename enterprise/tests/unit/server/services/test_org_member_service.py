"""Tests for OrgMemberService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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
