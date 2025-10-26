"""
Store class for managing organizations.
"""

import uuid
from typing import Optional
from uuid import UUID

from server.constants import ORG_SETTINGS_VERSION, get_default_litellm_model
from sqlalchemy.orm import joinedload
from storage.database import session_maker
from storage.org import Org
from storage.user import User

from openhands.core.logger import openhands_logger as logger
from openhands.storage.data_models.settings import Settings


class OrgStore:
    """Store for managing organizations."""

    @staticmethod
    def create_org(
        kwargs: dict,
    ) -> Org:
        """Create a new organization."""
        with session_maker() as session:
            org = Org(**kwargs)
            org.org_version = ORG_SETTINGS_VERSION
            org.default_llm_model = get_default_litellm_model()
            session.add(org)
            session.commit()
            session.refresh(org)
            return org

    @staticmethod
    def get_org_by_id(org_id: UUID) -> Org | None:
        """Get organization by ID."""
        with session_maker() as session:
            return session.query(Org).filter(Org.id == org_id).first()

    @staticmethod
    def get_current_org_from_keycloak_user_id(keycloak_user_id: str) -> Org | None:
        with session_maker() as session:
            user = (
                session.query(User)
                .options(joinedload(User.org_members))
                .filter(User.id == uuid.UUID(keycloak_user_id))
                .first()
            )
            if not user:
                logger.warning(f'User not found for ID {keycloak_user_id}')
                return None
            org_id = user.current_org_id
            org = session.query(Org).filter(Org.id == org_id).first()
            if not org:
                logger.warning(
                    f'Org not found for ID {org_id} as the current org for user {keycloak_user_id}'
                )
                return None
            return org

    @staticmethod
    def get_org_by_name(name: str) -> Org | None:
        """Get organization by name."""
        with session_maker() as session:
            return session.query(Org).filter(Org.name == name).first()

    @staticmethod
    def list_orgs() -> list[Org]:
        """List all organizations."""
        with session_maker() as session:
            orgs = session.query(Org).all()
            return orgs

    @staticmethod
    def update_org(
        org_id: UUID,
        kwargs: dict,
    ) -> Optional[Org]:
        """Update organization details."""
        with session_maker() as session:
            org = session.query(Org).filter(Org.id == org_id).first()
            if not org:
                return None

            if 'org_id' in kwargs:
                kwargs.pop('org_id')
            for key, value in kwargs.items():
                if hasattr(org, key):
                    setattr(org, key, value)

            session.commit()
            session.refresh(org)
            return org

    @staticmethod
    def get_kwargs_from_settings(settings: Settings):
        kwargs = {
            c.name: getattr(settings, normalized)
            for c in Org.__table__.columns
            if (
                normalized := c.name.removeprefix('_default_')
                .removeprefix('default_')
                .lstrip('_')
            )
            and hasattr(settings, normalized)
        }
        return kwargs
