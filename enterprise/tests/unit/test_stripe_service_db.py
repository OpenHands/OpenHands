"""
This test file verifies that the stripe_service functions properly use the database
to store and retrieve customer IDs.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from integrations.stripe_service import (
    find_customer_id_by_user_id,
    find_or_create_customer,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.stripe_customer import Base as StripeCustomerBase
from storage.stripe_customer import StripeCustomer
from storage.user_settings import Base as UserBase


@pytest.fixture
def engine():
    engine = create_engine('sqlite:///:memory:')

    UserBase.metadata.create_all(engine)
    StripeCustomerBase.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_maker(engine):
    return sessionmaker(bind=engine)


@pytest.mark.asyncio
async def test_find_customer_id_by_user_id_checks_db_first(session_maker):
    """Test that find_customer_id_by_user_id checks the database first"""

    # Set up the mock for the database query result
    with session_maker() as session:
        session.add(
            StripeCustomer(
                keycloak_user_id='test-user-id',
                stripe_customer_id='cus_test123',
            )
        )
        session.commit()

    with patch('integrations.stripe_service.session_maker', session_maker):
        # Call the function
        result = await find_customer_id_by_user_id('test-user-id')

        # Verify the result
        assert result == 'cus_test123'


@pytest.mark.asyncio
async def test_find_customer_id_by_user_id_falls_back_to_stripe(session_maker):
    """Test that find_customer_id_by_user_id falls back to Stripe if not found in the database"""

    # Set up the mock for stripe.Customer.search_async
    mock_customer = stripe.Customer(id='cus_test123')
    mock_search = AsyncMock(return_value=MagicMock(data=[mock_customer]))

    with (
        patch('integrations.stripe_service.session_maker', session_maker),
        patch('stripe.Customer.search_async', mock_search),
    ):
        # Call the function
        result = await find_customer_id_by_user_id('test-user-id')

        # Verify the result
        assert result == 'cus_test123'

    # Verify that Stripe was searched
    mock_search.assert_called_once()
    assert "metadata['user_id']:'test-user-id'" in mock_search.call_args[1]['query']


@pytest.mark.asyncio
async def test_create_customer_stores_id_in_db(session_maker):
    """Test that create_customer stores the customer ID in the database"""

    # Set up the mock for stripe.Customer.search_async
    mock_search = AsyncMock(return_value=MagicMock(data=[]))
    mock_create_async = AsyncMock(return_value=stripe.Customer(id='cus_test123'))

    with (
        patch('integrations.stripe_service.session_maker', session_maker),
        patch('stripe.Customer.search_async', mock_search),
        patch('stripe.Customer.create_async', mock_create_async),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'testy@tester.com'}),
        ),
    ):
        # Call the function
        result = await find_or_create_customer('test-user-id')

    # Verify the result
    assert result == 'cus_test123'

    # Verify that the stripe customer was stored in the db
    with session_maker() as session:
        customer = session.query(StripeCustomer).first()
        assert customer.id > 0
        assert customer.keycloak_user_id == 'test-user-id'
        assert customer.stripe_customer_id == 'cus_test123'
        assert customer.created_at is not None
        assert customer.updated_at is not None
