from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from fastapi import HTTPException, Request, status
from httpx import HTTPStatusError, Response
from integrations.stripe_service import has_payment_method
from server.routes.billing import (
    CreateBillingSessionResponse,
    CreateCheckoutSessionRequest,
    GetCreditsResponse,
    cancel_callback,
    cancel_subscription,
    create_checkout_session,
    create_subscription_checkout_session,
    get_credits,
    success_callback,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import URL
from storage.billing_session_type import BillingSessionType
from storage.stripe_customer import Base as StripeCustomerBase


@pytest.fixture
def engine():
    engine = create_engine('sqlite:///:memory:')
    StripeCustomerBase.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_maker(engine):
    return sessionmaker(bind=engine)


@pytest.fixture
def mock_request():
    """Create a mock request object with proper URL structure for testing."""
    return Request(
        scope={
            'type': 'http',
            'path': '/api/billing/test',
            'server': ('test.com', 80),
        }
    )


@pytest.fixture
def mock_checkout_request():
    """Create a mock request object for checkout session tests."""
    request = Request(
        scope={
            'type': 'http',
            'path': '/api/billing/create-checkout-session',
            'server': ('test.com', 80),
        }
    )
    request._base_url = URL('http://test.com/')
    return request


@pytest.fixture
def mock_subscription_request():
    """Create a mock request object for subscription checkout session tests."""
    request = Request(
        scope={
            'type': 'http',
            'path': '/api/billing/subscription-checkout-session',
            'server': ('test.com', 80),
        }
    )
    request._base_url = URL('http://test.com/')
    return request


@pytest.mark.asyncio
async def test_get_credits_lite_llm_error():
    mock_request = Request(scope={'type': 'http', 'state': {'user_id': 'mock_user'}})

    mock_response = Response(
        status_code=500, json={'error': 'Internal Server Error'}, request=MagicMock()
    )
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch('integrations.stripe_service.STRIPE_API_KEY', 'mock_key'):
        with patch('httpx.AsyncClient', return_value=mock_client):
            with pytest.raises(HTTPStatusError) as exc_info:
                await get_credits(mock_request)
            assert (
                exc_info.value.response.status_code
                == status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@pytest.mark.asyncio
async def test_get_credits_success():
    mock_response = Response(
        status_code=200,
        json={'user_info': {'max_budget': 100.00, 'spend': 25.50}},
        request=MagicMock(),
    )
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with (
        patch('integrations.stripe_service.STRIPE_API_KEY', 'mock_key'),
        patch('httpx.AsyncClient', return_value=mock_client),
    ):
        with patch('server.routes.billing.session_maker') as mock_session_maker:
            mock_db_session = MagicMock()
            mock_db_session.query.return_value.filter.return_value.first.return_value = MagicMock(
                billing_margin=4
            )
            mock_session_maker.return_value.__enter__.return_value = mock_db_session

            result = await get_credits('mock_user')

            assert isinstance(result, GetCreditsResponse)
            assert result.credits == Decimal(
                '74.50'
            )  # 100.00 - 25.50 = 74.50 (no billing margin applied)
            mock_client.__aenter__.return_value.get.assert_called_once_with(
                'https://llm-proxy.app.all-hands.dev/user/info?user_id=mock_user',
                headers={'x-goog-api-key': None},
            )


@pytest.mark.asyncio
async def test_create_checkout_session_stripe_error(
    session_maker, mock_checkout_request
):
    """Test handling of Stripe API errors."""

    mock_customer = stripe.Customer(
        id='mock-customer', metadata={'user_id': 'mock-user'}
    )
    mock_customer_create = AsyncMock(return_value=mock_customer)
    with (
        pytest.raises(Exception, match='Stripe API Error'),
        patch('stripe.Customer.create_async', mock_customer_create),
        patch(
            'stripe.Customer.search_async', AsyncMock(return_value=MagicMock(data=[]))
        ),
        patch(
            'stripe.checkout.Session.create_async',
            AsyncMock(side_effect=Exception('Stripe API Error')),
        ),
        patch('integrations.stripe_service.session_maker', session_maker),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'testy@tester.com'}),
        ),
        patch('server.routes.billing.validate_saas_environment'),
    ):
        await create_checkout_session(
            CreateCheckoutSessionRequest(amount=25), mock_checkout_request, 'mock_user'
        )


@pytest.mark.asyncio
async def test_create_checkout_session_success(session_maker, mock_checkout_request):
    """Test successful creation of checkout session."""

    mock_session = MagicMock()
    mock_session.url = 'https://checkout.stripe.com/test-session'
    mock_session.id = 'test_session_id'
    mock_create = AsyncMock(return_value=mock_session)
    mock_create.return_value = mock_session

    mock_customer = stripe.Customer(
        id='mock-customer', metadata={'user_id': 'mock-user'}
    )
    mock_customer_create = AsyncMock(return_value=mock_customer)
    with (
        patch('stripe.Customer.create_async', mock_customer_create),
        patch(
            'stripe.Customer.search_async', AsyncMock(return_value=MagicMock(data=[]))
        ),
        patch('stripe.checkout.Session.create_async', mock_create),
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch('integrations.stripe_service.session_maker', session_maker),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'testy@tester.com'}),
        ),
        patch('server.routes.billing.validate_saas_environment'),
    ):
        mock_db_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_db_session

        result = await create_checkout_session(
            CreateCheckoutSessionRequest(amount=25), mock_checkout_request, 'mock_user'
        )

        assert isinstance(result, CreateBillingSessionResponse)
        assert result.redirect_url == 'https://checkout.stripe.com/test-session'

        # Verify Stripe session creation parameters
        mock_create.assert_called_once_with(
            customer='mock-customer',
            line_items=[
                {
                    'price_data': {
                        'unit_amount': 2500,
                        'currency': 'usd',
                        'product_data': {
                            'name': 'OpenHands Credits',
                            'tax_code': 'txcd_10000000',
                        },
                        'tax_behavior': 'exclusive',
                    },
                    'quantity': 1,
                }
            ],
            mode='payment',
            payment_method_types=['card'],
            saved_payment_method_options={'payment_method_save': 'enabled'},
            success_url='http://test.com/api/billing/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://test.com/api/billing/cancel?session_id={CHECKOUT_SESSION_ID}',
        )

        # Verify database session creation
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_success_callback_session_not_found():
    """Test success callback when billing session is not found."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._base_url = URL('http://test.com/')

    with patch('server.routes.billing.session_maker') as mock_session_maker:
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = None
        mock_session_maker.return_value.__enter__.return_value = mock_db_session
        with pytest.raises(HTTPException) as exc_info:
            await success_callback('test_session_id', mock_request)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        mock_db_session.merge.assert_not_called()
        mock_db_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_success_callback_stripe_incomplete():
    """Test success callback when Stripe session is not complete."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._base_url = URL('http://test.com/')

    mock_billing_session = MagicMock()
    mock_billing_session.status = 'in_progress'
    mock_billing_session.user_id = 'mock_user'
    mock_billing_session.billing_session_type = BillingSessionType.DIRECT_PAYMENT.value

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
    ):
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_billing_session
        mock_session_maker.return_value.__enter__.return_value = mock_db_session

        mock_stripe_retrieve.return_value = MagicMock(status='pending')

        with pytest.raises(HTTPException) as exc_info:
            await success_callback('test_session_id', mock_request)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        mock_db_session.merge.assert_not_called()
        mock_db_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_success_callback_success():
    """Test successful payment completion and credit update."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._base_url = URL('http://test.com/')

    mock_billing_session = MagicMock()
    mock_billing_session.status = 'in_progress'
    mock_billing_session.user_id = 'mock_user'
    mock_billing_session.billing_session_type = BillingSessionType.DIRECT_PAYMENT.value

    mock_lite_llm_response = Response(
        status_code=200,
        json={'user_info': {'max_budget': 100.00, 'spend': 25.50}},
        request=MagicMock(),
    )
    mock_lite_llm_update_response = Response(
        status_code=200, json={}, request=MagicMock()
    )

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
        patch('httpx.AsyncClient') as mock_client,
    ):
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_billing_session
        mock_user_settings = MagicMock(billing_margin=None)
        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            mock_user_settings
        )
        mock_session_maker.return_value.__enter__.return_value = mock_db_session

        mock_stripe_retrieve.return_value = MagicMock(
            status='complete',
            amount_subtotal=2500,
        )  # $25.00 in cents

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value.get.return_value = (
            mock_lite_llm_response
        )
        mock_client_instance.__aenter__.return_value.post.return_value = (
            mock_lite_llm_update_response
        )
        mock_client.return_value = mock_client_instance

        response = await success_callback('test_session_id', mock_request)

        assert response.status_code == 302
        assert (
            response.headers['location']
            == 'http://test.com/settings/billing?checkout=success'
        )

        # Verify LiteLLM API calls
        mock_client_instance.__aenter__.return_value.get.assert_called_once()
        mock_client_instance.__aenter__.return_value.post.assert_called_once_with(
            'https://llm-proxy.app.all-hands.dev/user/update',
            headers={'x-goog-api-key': None},
            json={
                'user_id': 'mock_user',
                'max_budget': 125,
            },  # 100 + (25.00 from Stripe)
        )

        # Verify database updates
        assert mock_billing_session.status == 'completed'
        mock_db_session.merge.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_success_callback_lite_llm_error():
    """Test handling of LiteLLM API errors during success callback."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._base_url = URL('http://test.com/')

    mock_billing_session = MagicMock()
    mock_billing_session.status = 'in_progress'
    mock_billing_session.user_id = 'mock_user'
    mock_billing_session.billing_session_type = BillingSessionType.DIRECT_PAYMENT.value

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
        patch('httpx.AsyncClient') as mock_client,
    ):
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_billing_session
        mock_session_maker.return_value.__enter__.return_value = mock_db_session

        mock_stripe_retrieve.return_value = MagicMock(
            status='complete', amount_total=2500
        )

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value.get.side_effect = Exception(
            'LiteLLM API Error'
        )
        mock_client.return_value = mock_client_instance

        with pytest.raises(Exception, match='LiteLLM API Error'):
            await success_callback('test_session_id', mock_request)

        # Verify no database updates occurred
        assert mock_billing_session.status == 'in_progress'
        mock_db_session.merge.assert_not_called()
        mock_db_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_callback_session_not_found():
    """Test cancel callback when billing session is not found."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._base_url = URL('http://test.com/')

    with patch('server.routes.billing.session_maker') as mock_session_maker:
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = None
        mock_session_maker.return_value.__enter__.return_value = mock_db_session

        response = await cancel_callback('test_session_id', mock_request)
        assert response.status_code == 302
        assert (
            response.headers['location'] == 'http://test.com/settings?checkout=cancel'
        )

        # Verify no database updates occurred
        mock_db_session.merge.assert_not_called()
        mock_db_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_callback_success():
    """Test successful cancellation of billing session."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._base_url = URL('http://test.com/')

    mock_billing_session = MagicMock()
    mock_billing_session.status = 'in_progress'

    with patch('server.routes.billing.session_maker') as mock_session_maker:
        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_billing_session
        mock_session_maker.return_value.__enter__.return_value = mock_db_session

        response = await cancel_callback('test_session_id', mock_request)

        assert response.status_code == 302
        assert (
            response.headers['location'] == 'http://test.com/settings?checkout=cancel'
        )

        # Verify database updates
        assert mock_billing_session.status == 'cancelled'
        mock_db_session.merge.assert_called_once()
        mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_has_payment_method_with_payment_method():
    """Test has_payment_method returns True when user has a payment method."""
    with (
        patch('integrations.stripe_service.session_maker') as mock_session_maker,
        patch(
            'stripe.Customer.list_payment_methods_async',
            AsyncMock(return_value=MagicMock(data=[MagicMock()])),
        ) as mock_list_payment_methods,
    ):
        # Setup mock session
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            MagicMock(stripe_customer_id='cus_test123')
        )

        result = await has_payment_method('mock_user')
        assert result is True
        mock_list_payment_methods.assert_called_once_with('cus_test123')


@pytest.mark.asyncio
async def test_has_payment_method_without_payment_method():
    """Test has_payment_method returns False when user has no payment method."""
    with (
        patch('integrations.stripe_service.session_maker') as mock_session_maker,
        patch(
            'stripe.Customer.list_payment_methods_async',
            AsyncMock(return_value=MagicMock(data=[])),
        ) as mock_list_payment_methods,
    ):
        # Setup mock session
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            MagicMock(stripe_customer_id='cus_test123')
        )

        result = await has_payment_method('mock_user')
        assert result is False
        mock_list_payment_methods.assert_called_once_with('cus_test123')


@pytest.mark.asyncio
async def test_cancel_subscription_success():
    """Test successful subscription cancellation."""
    from datetime import UTC, datetime

    from storage.subscription_access import SubscriptionAccess

    # Mock active subscription
    mock_subscription_access = SubscriptionAccess(
        id=1,
        status='ACTIVE',
        user_id='test_user',
        start_at=datetime.now(UTC),
        end_at=datetime.now(UTC),
        amount_paid=2000,
        stripe_invoice_payment_id='pi_test',
        stripe_subscription_id='sub_test123',
        cancelled_at=None,
    )

    # Mock Stripe subscription response
    mock_stripe_subscription = MagicMock()
    mock_stripe_subscription.cancel_at_period_end = True

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch(
            'stripe.Subscription.modify_async',
            AsyncMock(return_value=mock_stripe_subscription),
        ) as mock_stripe_modify,
    ):
        # Setup mock session
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = mock_subscription_access

        # Call the function
        result = await cancel_subscription('test_user')

        # Verify Stripe API was called
        mock_stripe_modify.assert_called_once_with(
            'sub_test123', cancel_at_period_end=True
        )

        # Verify database was updated
        assert mock_subscription_access.cancelled_at is not None
        mock_session.merge.assert_called_once_with(mock_subscription_access)
        mock_session.commit.assert_called_once()

        # Verify response
        assert result.status_code == 200


@pytest.mark.asyncio
async def test_cancel_subscription_no_active_subscription():
    """Test cancellation when no active subscription exists."""
    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
    ):
        # Setup mock session with no subscription found
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = None

        # Call the function and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await cancel_subscription('test_user')

        assert exc_info.value.status_code == 404
        assert 'No active subscription found' in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_cancel_subscription_missing_stripe_id():
    """Test cancellation when subscription has no Stripe ID."""
    from datetime import UTC, datetime

    from storage.subscription_access import SubscriptionAccess

    # Mock subscription without Stripe ID
    mock_subscription_access = SubscriptionAccess(
        id=1,
        status='ACTIVE',
        user_id='test_user',
        start_at=datetime.now(UTC),
        end_at=datetime.now(UTC),
        amount_paid=2000,
        stripe_invoice_payment_id='pi_test',
        stripe_subscription_id=None,  # Missing Stripe ID
        cancelled_at=None,
    )

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
    ):
        # Setup mock session
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = mock_subscription_access

        # Call the function and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await cancel_subscription('test_user')

        assert exc_info.value.status_code == 400
        assert 'missing Stripe subscription ID' in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_cancel_subscription_stripe_error():
    """Test cancellation when Stripe API fails."""
    from datetime import UTC, datetime

    from storage.subscription_access import SubscriptionAccess

    # Mock active subscription
    mock_subscription_access = SubscriptionAccess(
        id=1,
        status='ACTIVE',
        user_id='test_user',
        start_at=datetime.now(UTC),
        end_at=datetime.now(UTC),
        amount_paid=2000,
        stripe_invoice_payment_id='pi_test',
        stripe_subscription_id='sub_test123',
        cancelled_at=None,
    )

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch(
            'stripe.Subscription.modify_async',
            AsyncMock(side_effect=stripe.StripeError('API Error')),
        ),
    ):
        # Setup mock session
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = mock_subscription_access

        # Call the function and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await cancel_subscription('test_user')

        assert exc_info.value.status_code == 500
        assert 'Failed to cancel subscription' in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_subscription_checkout_session_duplicate_prevention(
    mock_subscription_request,
):
    """Test that creating a subscription when user already has active subscription raises error."""
    from datetime import UTC, datetime

    from storage.subscription_access import SubscriptionAccess

    # Mock active subscription
    mock_subscription_access = SubscriptionAccess(
        id=1,
        status='ACTIVE',
        user_id='test_user',
        start_at=datetime.now(UTC),
        end_at=datetime.now(UTC),
        amount_paid=2000,
        stripe_invoice_payment_id='pi_test',
        stripe_subscription_id='sub_test123',
        cancelled_at=None,
    )

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch('server.routes.billing.validate_saas_environment'),
    ):
        # Setup mock session to return existing active subscription
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = mock_subscription_access

        # Call the function and expect HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_subscription_checkout_session(
                mock_subscription_request, user_id='test_user'
            )

        assert exc_info.value.status_code == 400
        assert (
            'user already has an active subscription'
            in str(exc_info.value.detail).lower()
        )


@pytest.mark.asyncio
async def test_create_subscription_checkout_session_allows_after_cancellation(
    mock_subscription_request,
):
    """Test that creating a subscription is allowed when previous subscription was cancelled."""

    mock_session_obj = MagicMock()
    mock_session_obj.url = 'https://checkout.stripe.com/test-session'
    mock_session_obj.id = 'test_session_id'

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch(
            'integrations.stripe_service.find_or_create_customer',
            AsyncMock(return_value='cus_test123'),
        ),
        patch(
            'stripe.checkout.Session.create_async',
            AsyncMock(return_value=mock_session_obj),
        ),
        patch(
            'server.routes.billing.SUBSCRIPTION_PRICE_DATA',
            {'MONTHLY_SUBSCRIPTION': {'unit_amount': 2000}},
        ),
        patch('server.routes.billing.validate_saas_environment'),
    ):
        # Setup mock session - the query should return None because cancelled subscriptions are filtered out
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = None

        # Should succeed
        result = await create_subscription_checkout_session(
            mock_subscription_request, user_id='test_user'
        )

        assert isinstance(result, CreateBillingSessionResponse)
        assert result.redirect_url == 'https://checkout.stripe.com/test-session'


@pytest.mark.asyncio
async def test_create_subscription_checkout_session_success_no_existing(
    mock_subscription_request,
):
    """Test successful subscription creation when no existing subscription."""

    mock_session_obj = MagicMock()
    mock_session_obj.url = 'https://checkout.stripe.com/test-session'
    mock_session_obj.id = 'test_session_id'

    with (
        patch('server.routes.billing.session_maker') as mock_session_maker,
        patch(
            'integrations.stripe_service.find_or_create_customer',
            AsyncMock(return_value='cus_test123'),
        ),
        patch(
            'stripe.checkout.Session.create_async',
            AsyncMock(return_value=mock_session_obj),
        ),
        patch(
            'server.routes.billing.SUBSCRIPTION_PRICE_DATA',
            {'MONTHLY_SUBSCRIPTION': {'unit_amount': 2000}},
        ),
        patch('server.routes.billing.validate_saas_environment'),
    ):
        # Setup mock session to return no existing subscription
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value.first.return_value = None

        # Should succeed
        result = await create_subscription_checkout_session(
            mock_subscription_request, user_id='test_user'
        )

        assert isinstance(result, CreateBillingSessionResponse)
        assert result.redirect_url == 'https://checkout.stripe.com/test-session'
