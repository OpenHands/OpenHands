"""
This test file verifies that the stripe_service functions properly use the database
to store and retrieve customer IDs.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from integrations.stripe_service import (
    find_customer_id_by_user_id,
    find_or_create_customer_by_user_id,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.base import Base
from storage.org import Org
from storage.org_member import OrgMember
from storage.role import Role
from storage.stripe_customer import StripeCustomer
from storage.user import User


@pytest.fixture
def engine():
    engine = create_engine('sqlite:///:memory:')
    # Create all tables using the unified Base
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_maker(engine):
    return sessionmaker(bind=engine)


@pytest.fixture
def test_org_and_user(session_maker):
    """Create a test org and user for use in tests."""
    test_user_id = uuid.uuid4()
    test_org_id = uuid.uuid4()

    with session_maker() as session:
        # Create role first
        role = Role(name='test-role', rank=1)
        session.add(role)
        session.flush()

        # Create org
        org = Org(id=test_org_id, name='test-org', contact_email='testy@tester.com')
        session.add(org)
        session.flush()

        # Create user with current_org_id
        user = User(id=test_user_id, current_org_id=test_org_id, role_id=role.id)
        session.add(user)
        session.flush()

        # Create org member relationship
        org_member = OrgMember(
            org_id=test_org_id,
            user_id=test_user_id,
            role_id=role.id,
            llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    return test_user_id, test_org_id


@pytest.mark.asyncio
async def test_find_customer_id_by_user_id_checks_db_first(
    session_maker, test_org_and_user
):
    """Test that find_customer_id_by_user_id checks the database first"""

    test_user_id, test_org_id = test_org_and_user

    # Set up the mock for the database query result
    with session_maker() as session:
        # Create stripe customer
        session.add(
            StripeCustomer(
                keycloak_user_id=str(test_user_id),
                org_id=test_org_id,
                stripe_customer_id='cus_test123',
            )
        )
        session.commit()

    # Create a mock org object to return from OrgStore
    mock_org = MagicMock()
    mock_org.id = test_org_id

    with (
        patch('integrations.stripe_service.session_maker', session_maker),
        patch('storage.org_store.session_maker', session_maker),
        patch('integrations.stripe_service.call_sync_from_async') as mock_call_sync,
    ):
        # Mock the call_sync_from_async to return the org
        mock_call_sync.return_value = mock_org

        # Call the function
        result = await find_customer_id_by_user_id(str(test_user_id))

        # Verify the result
        assert result == 'cus_test123'

        # Verify that call_sync_from_async was called with the correct function
        mock_call_sync.assert_called_once()


@pytest.mark.asyncio
async def test_find_customer_id_by_user_id_falls_back_to_stripe(
    session_maker, test_org_and_user
):
    """Test that find_customer_id_by_user_id falls back to Stripe if not found in the database"""

    test_user_id, test_org_id = test_org_and_user

    # Set up the mock for stripe.Customer.search_async
    mock_customer = stripe.Customer(id='cus_test123')
    mock_search = AsyncMock(return_value=MagicMock(data=[mock_customer]))

    # Create a mock org object to return from OrgStore
    mock_org = MagicMock()
    mock_org.id = test_org_id

    with (
        patch('integrations.stripe_service.session_maker', session_maker),
        patch('storage.org_store.session_maker', session_maker),
        patch('stripe.Customer.search_async', mock_search),
        patch('integrations.stripe_service.call_sync_from_async') as mock_call_sync,
    ):
        # Mock the call_sync_from_async to return the org
        mock_call_sync.return_value = mock_org

        # Call the function
        result = await find_customer_id_by_user_id(str(test_user_id))

        # Verify the result
        assert result == 'cus_test123'

    # Verify that Stripe was searched with the org_id
    mock_search.assert_called_once()
    assert (
        f"metadata['org_id']:'{str(test_org_id)}'" in mock_search.call_args[1]['query']
    )


@pytest.mark.asyncio
async def test_create_customer_stores_id_in_db(session_maker, test_org_and_user):
    """Test that create_customer stores the customer ID in the database"""

    test_user_id, test_org_id = test_org_and_user

    # Set up the mock for stripe.Customer.search_async and create_async
    mock_search = AsyncMock(return_value=MagicMock(data=[]))
    mock_create_async = AsyncMock(return_value=stripe.Customer(id='cus_test123'))

    # Create a mock org object to return from OrgStore
    mock_org = MagicMock()
    mock_org.id = test_org_id
    mock_org.contact_email = 'testy@tester.com'

    with (
        patch('integrations.stripe_service.session_maker', session_maker),
        patch('storage.org_store.session_maker', session_maker),
        patch('stripe.Customer.search_async', mock_search),
        patch('stripe.Customer.create_async', mock_create_async),
        patch('integrations.stripe_service.call_sync_from_async') as mock_call_sync,
    ):
        # Mock the call_sync_from_async to return the org
        mock_call_sync.return_value = mock_org

        # Call the function
        result = await find_or_create_customer_by_user_id(str(test_user_id))

    # Verify the result
    assert result == {'customer_id': 'cus_test123', 'org_id': str(test_org_id)}

    # Verify that the stripe customer was stored in the db
    with session_maker() as session:
        customer = session.query(StripeCustomer).first()
        assert customer.id > 0
        assert customer.keycloak_user_id == str(test_user_id)
        assert customer.org_id == test_org_id
        assert customer.stripe_customer_id == 'cus_test123'
        assert customer.created_at is not None
        assert customer.updated_at is not None
