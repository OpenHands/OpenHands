"""Tests for UserBase Protocol structural typing compatibility."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from openhands.analytics.user_base import UserBase


class TestUserBaseProtocol:
    """Tests verifying UserBase Protocol works with various implementations.

    UserBase uses Any types for SQLAlchemy compatibility, but these tests
    verify the Protocol's structural typing works correctly at runtime.
    """

    def test_protocol_with_uuid_types(self):
        """UserBase Protocol accepts a class using UUID for id and current_org_id."""

        @dataclass
        class UserWithUUID:
            id: UUID
            user_consents_to_analytics: bool | None
            current_org_id: UUID | None
            accepted_tos: datetime | None

        user = UserWithUUID(
            id=uuid4(),
            user_consents_to_analytics=True,
            current_org_id=uuid4(),
            accepted_tos=datetime.now(),
        )

        assert isinstance(user, UserBase)
        assert user.user_consents_to_analytics is True

    def test_protocol_with_str_types(self):
        """UserBase Protocol accepts a class using str for id and current_org_id."""

        @dataclass
        class UserWithStr:
            id: str
            user_consents_to_analytics: bool | None
            current_org_id: str | None
            accepted_tos: datetime | None

        user = UserWithStr(
            id='user-123',
            user_consents_to_analytics=False,
            current_org_id='org-456',
            accepted_tos=None,
        )

        assert isinstance(user, UserBase)
        assert user.user_consents_to_analytics is False
        assert user.accepted_tos is None

    def test_protocol_with_none_values(self):
        """UserBase Protocol accepts None for nullable fields."""

        @dataclass
        class UserWithNones:
            id: str
            user_consents_to_analytics: bool | None
            current_org_id: str | None
            accepted_tos: datetime | None

        user = UserWithNones(
            id='user-789',
            user_consents_to_analytics=None,
            current_org_id=None,
            accepted_tos=None,
        )

        assert isinstance(user, UserBase)
        assert user.user_consents_to_analytics is None
        assert user.current_org_id is None

    def test_protocol_with_extra_attributes(self):
        """UserBase Protocol accepts classes with additional attributes."""

        @dataclass
        class UserWithExtras:
            id: UUID
            user_consents_to_analytics: bool | None
            current_org_id: UUID | None
            accepted_tos: datetime | None
            email: str  # Extra attribute
            role: str  # Extra attribute

        user = UserWithExtras(
            id=uuid4(),
            user_consents_to_analytics=True,
            current_org_id=uuid4(),
            accepted_tos=datetime.now(),
            email='test@example.com',
            role='admin',
        )

        assert isinstance(user, UserBase)

    def test_protocol_rejects_missing_attributes(self):
        """UserBase Protocol rejects classes missing required attributes."""

        @dataclass
        class IncompleteUser:
            id: str
            user_consents_to_analytics: bool | None
            # Missing: current_org_id, accepted_tos

        user = IncompleteUser(
            id='user-incomplete',
            user_consents_to_analytics=True,
        )

        assert not isinstance(user, UserBase)

    def test_protocol_with_mock_object(self):
        """UserBase Protocol works with mock objects (as used in tests)."""
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_consents_to_analytics = True
        mock_user.current_org_id = 'org-mock'
        mock_user.accepted_tos = datetime.now()

        # MagicMock has all attributes, so it satisfies the protocol
        assert isinstance(mock_user, UserBase)
        assert mock_user.user_consents_to_analytics is True

    def test_protocol_with_any_typed_class(self):
        """UserBase Protocol works with Any-typed attributes (SQLAlchemy compatibility)."""

        @dataclass
        class UserWithAny:
            id: Any
            user_consents_to_analytics: Any
            current_org_id: Any
            accepted_tos: Any

        user = UserWithAny(
            id=uuid4(),
            user_consents_to_analytics=True,
            current_org_id='org-any',
            accepted_tos=datetime.now(),
        )

        assert isinstance(user, UserBase)
