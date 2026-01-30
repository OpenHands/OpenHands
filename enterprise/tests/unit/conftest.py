import uuid
from datetime import datetime
from uuid import UUID

import pytest
from server.constants import ORG_SETTINGS_VERSION
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.base import Base

# Anything not loaded here may not have a table created for it.
from storage.billing_session import BillingSession
from storage.conversation_work import ConversationWork
from storage.device_code import DeviceCode  # noqa: F401
from storage.feedback import Feedback
from storage.github_app_installation import GithubAppInstallation
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import (
    StoredConversationMetadataSaas,
)
from storage.stored_offline_token import StoredOfflineToken
from storage.stripe_customer import StripeCustomer
from storage.user import User


@pytest.fixture
def engine():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_maker(engine):
    return sessionmaker(bind=engine)


def add_minimal_fixtures(session_maker):
    with session_maker() as session:
        session.add(
            BillingSession(
                id='mock-billing-session-id',
                user_id='mock-user-id',
                status='completed',
                price=20,
                price_code='NA',
                created_at=datetime.fromisoformat('2025-03-03'),
                updated_at=datetime.fromisoformat('2025-03-04'),
            )
        )
        session.add(
            Feedback(
                id='mock-feedback-id',
                version='1.0',
                email='user@all-hands.dev',
                polarity='positive',
                permissions='public',
                trajectory=[],
            )
        )
        session.add(
            GithubAppInstallation(
                installation_id='mock-installation-id',
                encrypted_token='',
                created_at=datetime.fromisoformat('2025-03-05'),
                updated_at=datetime.fromisoformat('2025-03-06'),
            )
        )
        session.add(
            StoredConversationMetadata(
                conversation_id='mock-conversation-id',
                created_at=datetime.fromisoformat('2025-03-07'),
                last_updated_at=datetime.fromisoformat('2025-03-08'),
                accumulated_cost=5.25,
                prompt_tokens=500,
                completion_tokens=250,
                total_tokens=750,
            )
        )
        session.add(
            StoredConversationMetadataSaas(
                conversation_id='mock-conversation-id',
                user_id=UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                org_id=UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
            )
        )
        session.add(
            StoredOfflineToken(
                user_id='mock-user-id',
                offline_token='mock-offline-token',
                created_at=datetime.fromisoformat('2025-03-07'),
                updated_at=datetime.fromisoformat('2025-03-08'),
            )
        )
        session.add(
            Org(
                id=uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                name='mock-org',
                org_version=ORG_SETTINGS_VERSION,
                enable_default_condenser=True,
                enable_proactive_conversation_starters=True,
            )
        )
        session.add(
            Role(
                id=1,
                name='admin',
                rank=1,
            )
        )
        session.add(
            User(
                id=uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                current_org_id=uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                user_consents_to_analytics=True,
            )
        )
        session.add(
            OrgMember(
                org_id=uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                user_id=uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081'),
                role_id=1,
                llm_api_key='mock-api-key',
                status='active',
            )
        )
        session.add(
            StripeCustomer(
                keycloak_user_id='mock-user-id',
                stripe_customer_id='mock-stripe-customer-id',
                created_at=datetime.fromisoformat('2025-03-09'),
                updated_at=datetime.fromisoformat('2025-03-10'),
            )
        )
        session.add(
            ConversationWork(
                conversation_id='mock-conversation-id',
                user_id='mock-user-id',
                created_at=datetime.fromisoformat('2025-03-07'),
                updated_at=datetime.fromisoformat('2025-03-08'),
            )
        )
        session.commit()


@pytest.fixture
def session_maker_with_minimal_fixtures(engine):
    session_maker = sessionmaker(bind=engine)
    add_minimal_fixtures(session_maker)
    return session_maker
