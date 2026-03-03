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
from storage.stripe_customer import StripeCustomer


def add_test_org_and_user(session_maker):
    """Create a test org and user for use in tests."""
    test_user_id = uuid.uuid4()
    test_org_id = uuid.uuid4()

    # Import here to avoid circular imports
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.user import User

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
    async_session_maker, session_maker_with_minimal_fixtures
):
    """Test that find_customer_id_by_user_id checks the database first"""

    # Add test org and user to the db
    test_user_id, test_org_id = add_test_org_and_user(
        session_maker_with_minimal_fixtures
    )

    # Create stripe customer in the db
    async with async_session_maker() as session:
        session.add(
            StripeCustomer(
                keycloak_user_id=str(test_user_id),
                org_id=test_org_id,
                stripe_customer_id='cus_test123',
            )
        )
        await session.commit()

    # Create a mock org object to return from OrgStore
    mock_org = MagicMock()
    mock_org.id = test_org_id

    with (
        patch('integrations.stripe_service.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
        patch(
            'integrations.stripe_service.OrgStore.get_current_org_from_keycloak_user_id',
            new_callable=AsyncMock,
        ) as mock_get_org,
    ):
        # Mock the async method to return the org
        mock_get_org.return_value = mock_org

        # Call the function
        result = await find_customer_id_by_user_id(str(test_user_id))

        # Verify the result
        assert result == 'cus_test123'

        # Verify that OrgStore.get_current_org_from_keycloak_user_id was called
        mock_get_org.assert_called_once_with(str(test_user_id))


@pytest.mark.asyncio
async def test_find_customer_id_by_user_id_falls_back_to_stripe(
    async_session_maker, session_maker_with_minimal_fixtures
):
    """Test that find_customer_id_by_user_id falls back to Stripe if not found in the database"""

    # Add test org and user to the db
    test_user_id, test_org_id = add_test_org_and_user(
        session_maker_with_minimal_fixtures
    )

    # Set up the mock for stripe.Customer.search_async
    mock_customer = stripe.Customer(id='cus_test123')
    mock_search = AsyncMock(return_value=MagicMock(data=[mock_customer]))

    # Create a mock org object to return from OrgStore
    mock_org = MagicMock()
    mock_org.id = test_org_id

    with (
        patch('integrations.stripe_service.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
        patch('stripe.Customer.search_async', mock_search),
        patch(
            'integrations.stripe_service.OrgStore.get_current_org_from_keycloak_user_id',
            new_callable=AsyncMock,
        ) as mock_get_org,
    ):
        # Mock the async method to return the org
        mock_get_org.return_value = mock_org

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
async def test_create_customer_stores_id_in_db(
    async_session_maker, session_maker_with_minimal_fixtures
):
    """Test that create_customer stores the customer ID in the database"""

    # Add test org and user to the db
    test_user_id, test_org_id = add_test_org_and_user(
        session_maker_with_minimal_fixtures
    )

    # Set up the mock for stripe.Customer.search_async and create_async
    mock_search = AsyncMock(return_value=MagicMock(data=[]))
    mock_create_async = AsyncMock(return_value=stripe.Customer(id='cus_test123'))

    # Create a mock org object to return from OrgStore
    mock_org = MagicMock()
    mock_org.id = test_org_id
    mock_org.contact_email = 'testy@tester.com'

    with (
        patch('integrations.stripe_service.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
        patch('stripe.Customer.search_async', mock_search),
        patch('stripe.Customer.create_async', mock_create_async),
        patch(
            'integrations.stripe_service.OrgStore.get_current_org_from_keycloak_user_id',
            new_callable=AsyncMock,
        ) as mock_get_org,
        patch(
            'integrations.stripe_service.find_customer_id_by_org_id',
            new_callable=AsyncMock,
        ) as mock_find_customer,
    ):
        # Mock the async method to return the org
        mock_get_org.return_value = mock_org
        # Mock find_customer_id_by_org_id to return None (force creation path)
        mock_find_customer.return_value = None

        # Call the function
        result = await find_or_create_customer_by_user_id(str(test_user_id))

    # Verify the result
    assert result == {'customer_id': 'cus_test123', 'org_id': str(test_org_id)}

    # Verify that the stripe customer was stored in the db
    async with async_session_maker() as session:
        from sqlalchemy import select

        stmt = select(StripeCustomer).where(
            StripeCustomer.keycloak_user_id == str(test_user_id)
        )
        result = await session.execute(stmt)
        customer = result.scalar_one_or_none()
        assert customer is not None
        assert customer.id > 0
        assert customer.keycloak_user_id == str(test_user_id)
        assert customer.org_id == test_org_id
        assert customer.stripe_customer_id == 'cus_test123'
        assert customer.created_at is not None
        assert customer.updated_at is not None
