"""Mock implementation of the stripe_service module for testing.
"""

from unittest.mock import AsyncMock, MagicMock

# Mock session maker
mock_db_session = MagicMock()
mock_session_maker = MagicMock()
mock_session_maker.return_value.__enter__.return_value = mock_db_session

# Mock stripe customer
mock_stripe_customer = MagicMock()
mock_stripe_customer.first.return_value = None
mock_db_session.query.return_value.filter.return_value = mock_stripe_customer

# Mock stripe search
mock_search_result = MagicMock()
mock_search_result.data = []
mock_search = AsyncMock(return_value=mock_search_result)

# Mock stripe create
mock_create_result = MagicMock()
mock_create_result.id = 'cus_test123'
mock_create = AsyncMock(return_value=mock_create_result)

# Mock stripe list payment methods
mock_payment_methods = MagicMock()
mock_payment_methods.data = []
mock_list_payment_methods = AsyncMock(return_value=mock_payment_methods)


# Mock functions
async def find_customer_id_by_user_id(user_id: str) -> str | None:
    """Mock implementation of find_customer_id_by_user_id"""
    # Check the database first
    with mock_session_maker() as session:
        stripe_customer = session.query(MagicMock()).filter(MagicMock()).first()
        if stripe_customer:
            return stripe_customer.stripe_customer_id

    # If that fails, fallback to stripe
    search_result = await mock_search(
        query=f"metadata['user_id']:'{user_id}'",
    )
    data = search_result.data
    if not data:
        return None
    return data[0].id


async def find_or_create_customer(user_id: str) -> str:
    """Mock implementation of find_or_create_customer"""
    customer_id = await find_customer_id_by_user_id(user_id)
    if customer_id:
        return customer_id

    # Create the customer in stripe
    customer = await mock_create(
        metadata={'user_id': user_id},
    )

    # Save the stripe customer in the local db
    with mock_session_maker() as session:
        session.add(MagicMock())
        session.commit()

    return customer.id


async def has_payment_method(user_id: str) -> bool:
    """Mock implementation of has_payment_method"""
    customer_id = await find_customer_id_by_user_id(user_id)
    if customer_id is None:
        return False
    await mock_list_payment_methods(
        customer_id,
    )
    # Always return True for testing
    return True
