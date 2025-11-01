"""
Test the Stripe customer ID validation and cleanup functionality.
This test verifies the fix for handling stale customer IDs in the database.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from integrations.stripe_service import (
    _validate_and_cleanup_customer_id,
    find_customer_id_by_user_id,
    has_payment_method,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.stored_settings import Base as StoredBase
from storage.stripe_customer import Base as StripeCustomerBase
from storage.stripe_customer import StripeCustomer
from storage.user_settings import Base as UserBase


@pytest.fixture
def engine():
    engine = create_engine('sqlite:///:memory:')
    StoredBase.metadata.create_all(engine)
    UserBase.metadata.create_all(engine)
    StripeCustomerBase.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_maker(engine):
    return sessionmaker(bind=engine)


@pytest.mark.asyncio
async def test_validate_and_cleanup_customer_id_valid():
    """Test validation with a valid customer ID."""
    with patch('stripe.Customer.retrieve_async') as mock_retrieve, \
         patch('integrations.stripe_service.STRIPE_API_KEY', 'sk_test_123'):
        
        mock_retrieve.return_value = MagicMock(id='cus_valid123')
        result = await _validate_and_cleanup_customer_id('user123', 'cus_valid123')
        assert result == 'cus_valid123'
        mock_retrieve.assert_called_once_with('cus_valid123')


@pytest.mark.asyncio
async def test_validate_and_cleanup_customer_id_invalid(session_maker):
    """Test validation with an invalid customer ID that gets cleaned up."""
    with patch('stripe.Customer.retrieve_async') as mock_retrieve, \
         patch('integrations.stripe_service.session_maker', session_maker), \
         patch('integrations.stripe_service.STRIPE_API_KEY', 'sk_test_123'):
        
        # Set up database with stale customer
        with session_maker() as session:
            session.add(
                StripeCustomer(
                    keycloak_user_id='user123',
                    stripe_customer_id='cus_invalid123',
                )
            )
            session.commit()
        
        # Mock Stripe error
        mock_retrieve.side_effect = stripe.InvalidRequestError(
            "No such customer: 'cus_invalid123'", None
        )
        
        result = await _validate_and_cleanup_customer_id('user123', 'cus_invalid123')
        assert result is None
        
        # Verify customer was removed from database
        with session_maker() as session:
            customer = (
                session.query(StripeCustomer)
                .filter(StripeCustomer.keycloak_user_id == 'user123')
                .first()
            )
            assert customer is None


@pytest.mark.asyncio
async def test_validate_and_cleanup_customer_id_no_api_key():
    """Test validation skips when no API key is configured."""
    with patch('integrations.stripe_service.STRIPE_API_KEY', None):
        result = await _validate_and_cleanup_customer_id('user123', 'cus_test123')
        assert result == 'cus_test123'


@pytest.mark.asyncio
async def test_find_customer_id_with_stale_customer_fallback(session_maker):
    """Test that find_customer_id_by_user_id handles stale customers and falls back to Stripe search."""
    with patch('integrations.stripe_service.session_maker', session_maker), \
         patch('integrations.stripe_service._validate_and_cleanup_customer_id') as mock_validate, \
         patch('stripe.Customer.search_async') as mock_search:
        
        # Set up database with stale customer
        with session_maker() as session:
            session.add(
                StripeCustomer(
                    keycloak_user_id='user123',
                    stripe_customer_id='cus_stale123',
                )
            )
            session.commit()
        
        # Mock validation returning None (stale customer cleaned up)
        mock_validate.return_value = None
        
        # Mock Stripe search returning a valid customer
        mock_search_result = MagicMock()
        mock_search_result.data = [MagicMock(id='cus_new123')]
        mock_search.return_value = mock_search_result
        
        result = await find_customer_id_by_user_id('user123')
        assert result == 'cus_new123'
        mock_validate.assert_called_once_with('user123', 'cus_stale123')


@pytest.mark.asyncio
async def test_has_payment_method_handles_invalid_customer():
    """Test that has_payment_method handles invalid customer errors gracefully."""
    with patch('integrations.stripe_service.find_customer_id_by_user_id') as mock_find, \
         patch('stripe.Customer.list_payment_methods_async') as mock_list:
        
        mock_find.return_value = 'cus_test123'
        mock_list.side_effect = stripe.InvalidRequestError(
            "No such customer: 'cus_test123'", None
        )
        
        result = await has_payment_method('user123')
        assert result is False