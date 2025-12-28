from datetime import datetime

import pytest
from server.constants import CURRENT_USER_SETTINGS_VERSION
from server.maintenance_task_processor.user_version_upgrade_processor import (
    UserVersionUpgradeProcessor,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.base import Base

# Anything not loaded here may not have a table created for it.
from storage.billing_session import BillingSession
from storage.conversation_work import ConversationWork
from storage.device_code import DeviceCode  # noqa: F401
from storage.feedback import Feedback
from storage.github_app_installation import GithubAppInstallation
from storage.maintenance_task import MaintenanceTask, MaintenanceTaskStatus
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_offline_token import StoredOfflineToken
from storage.stripe_customer import StripeCustomer
from storage.user_settings import UserSettings


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
                user_id='mock-user-id',
                created_at=datetime.fromisoformat('2025-03-07'),
                last_updated_at=datetime.fromisoformat('2025-03-08'),
                accumulated_cost=5.25,
                prompt_tokens=500,
                completion_tokens=250,
                total_tokens=750,
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
            StripeCustomer(
                keycloak_user_id='mock-user-id',
                stripe_customer_id='mock-stripe-customer-id',
                created_at=datetime.fromisoformat('2025-03-09'),
                updated_at=datetime.fromisoformat('2025-03-10'),
            )
        )
        session.add(
            UserSettings(
                keycloak_user_id='mock-user-id',
                user_consents_to_analytics=True,
                user_version=CURRENT_USER_SETTINGS_VERSION,
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
        maintenance_task = MaintenanceTask(
            status=MaintenanceTaskStatus.PENDING,
        )
        maintenance_task.set_processor(
            UserVersionUpgradeProcessor(
                user_ids=['mock-user-id'],
                created_at=datetime.fromisoformat('2025-03-07'),
                updated_at=datetime.fromisoformat('2025-03-08'),
            )
        )
        session.add(maintenance_task)
        session.commit()


@pytest.fixture
def session_maker_with_minimal_fixtures(engine):
    session_maker = sessionmaker(bind=engine)
    add_minimal_fixtures(session_maker)
    return session_maker
