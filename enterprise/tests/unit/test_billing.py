import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from fastapi import HTTPException, Request, status
from httpx import Response
from server.constants import ORG_SETTINGS_VERSION
from server.routes import billing
from server.routes.billing import (
    CreateBillingSessionResponse,
    CreateCheckoutSessionRequest,
    GetCreditsResponse,
    cancel_callback,
    create_checkout_session,
    create_customer_setup_session,
    get_credits,
    has_payment_method,
    success_callback,
)
from sqlalchemy import select
from starlette.datastructures import URL
from storage.billing_session import BillingSession
from storage.org import Org
from storage.user import User


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
    request._url = URL('http://test.com/')
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
    request._url = URL('http://test.com/')
    return request


@pytest.fixture
async def test_org(async_session_maker):
    """Create a test org in the database."""
    org_id = uuid.uuid4()
    async with async_session_maker() as session:
        org = Org(
            id=org_id,
            name=f'test-org-{org_id}',
            org_version=ORG_SETTINGS_VERSION,
            enable_proactive_conversation_starters=True,
        )
        session.add(org)
        await session.commit()
    return org


@pytest.fixture
async def test_user(async_session_maker, test_org):
    """Create a test user in the database linked to test_org."""
    user_id = uuid.uuid4()
    async with async_session_maker() as session:
        user = User(
            id=user_id,
            current_org_id=test_org.id,
            user_consents_to_analytics=True,
        )
        session.add(user)
        await session.commit()
    return user


@pytest.mark.asyncio
async def test_get_credits_lite_llm_error():
    with (
        patch('integrations.stripe_service.STRIPE_API_KEY', 'mock_key'),
        patch(
            'storage.user_store.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=MagicMock(current_org_id='mock_org_id'),
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.get_user_team_info',
            side_effect=Exception('LiteLLM API Error'),
        ),
    ):
        with pytest.raises(Exception, match='LiteLLM API Error'):
            await get_credits('mock_user')


@pytest.mark.asyncio
async def test_get_credits_success():
    mock_response = Response(
        status_code=200,
        json={
            'user_info': {
                'spend': 25.50,
                'max_budget_in_team': 100.00,
            }
        },
        request=MagicMock(),
    )
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with (
        patch('integrations.stripe_service.STRIPE_API_KEY', 'mock_key'),
        patch('httpx.AsyncClient', return_value=mock_client),
        patch(
            'storage.user_store.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=MagicMock(current_org_id='mock_org_id'),
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.get_user_team_info',
            return_value={
                'spend': 25.50,
                'max_budget_in_team': 100.00,
            },
        ),
    ):
        result = await get_credits('mock_user')

        assert isinstance(result, GetCreditsResponse)
        assert result.credits == Decimal('74.50')  # 100.00 - 25.50 = 74.50


@pytest.mark.asyncio
async def test_create_checkout_session_stripe_error(
    async_session_maker, mock_checkout_request, test_org
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
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('integrations.stripe_service.a_session_maker', async_session_maker),
        patch('storage.database.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
        patch(
            'storage.org_store.OrgStore.get_current_org_from_keycloak_user_id',
            return_value=test_org,
        ),
        patch(
            'server.auth.token_manager.TokenManager.get_user_info_from_user_id',
            AsyncMock(return_value={'email': 'testy@tester.com'}),
        ),
        patch('server.routes.billing.validate_billing_enabled'),
    ):
        await create_checkout_session(
            CreateCheckoutSessionRequest(amount=25), mock_checkout_request, 'mock_user'
        )


@pytest.mark.asyncio
async def test_create_checkout_session_success(
    async_session_maker, mock_checkout_request, test_org
):
    """Test successful creation of checkout session."""
    mock_session = MagicMock()
    mock_session.url = 'https://checkout.stripe.com/test-session'
    mock_session.id = 'test_session_id_checkout'
    mock_create = AsyncMock(return_value=mock_session)

    mock_customer_info = {'customer_id': 'mock-customer', 'org_id': test_org.id}

    with (
        patch('stripe.checkout.Session.create_async', mock_create),
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('integrations.stripe_service.a_session_maker', async_session_maker),
        patch(
            'integrations.stripe_service.find_or_create_customer_by_user_id',
            AsyncMock(return_value=mock_customer_info),
        ),
        patch('server.routes.billing.validate_billing_enabled'),
    ):
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
            success_url='https://test.com/api/billing/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='https://test.com/api/billing/cancel?session_id={CHECKOUT_SESSION_ID}',
        )

        # Verify database record was created
        async with async_session_maker() as session:
            result_db = await session.execute(
                select(BillingSession).where(
                    BillingSession.id == 'test_session_id_checkout'
                )
            )
            billing_session = result_db.scalar_one_or_none()
            assert billing_session is not None
            assert billing_session.user_id == 'mock_user'
            assert billing_session.org_id == test_org.id
            assert billing_session.status == 'in_progress'
            assert float(billing_session.price) == 25.0


@pytest.mark.asyncio
async def test_success_callback_session_not_found(async_session_maker):
    """Test success callback when billing session is not found."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    with (
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('stripe.checkout.Session.retrieve'),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await success_callback('nonexistent_session_id', mock_request)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_success_callback_stripe_incomplete(
    async_session_maker, test_org, test_user
):
    """Test success callback when Stripe session is not complete."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    session_id = 'test_incomplete_session'
    async with async_session_maker() as session:
        billing_session = BillingSession(
            id=session_id,
            user_id=str(test_user.id),
            org_id=test_org.id,
            status='in_progress',
            price=25,
            price_code='NA',
        )
        session.add(billing_session)
        await session.commit()

    with (
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
    ):
        mock_stripe_retrieve.return_value = MagicMock(status='pending')

        with pytest.raises(HTTPException) as exc_info:
            await success_callback(session_id, mock_request)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    # Verify no database update occurred
    async with async_session_maker() as session:
        result = await session.execute(
            select(BillingSession).where(BillingSession.id == session_id)
        )
        billing_session = result.scalar_one_or_none()
        assert billing_session.status == 'in_progress'


@pytest.mark.asyncio
async def test_success_callback_success(async_session_maker, test_org, test_user):
    """Test successful payment completion and credit update."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    session_id = 'test_success_session'
    async with async_session_maker() as session:
        billing_session = BillingSession(
            id=session_id,
            user_id=str(test_user.id),
            org_id=test_org.id,
            status='in_progress',
            price=25,
            price_code='NA',
        )
        session.add(billing_session)
        await session.commit()

    with (
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
        patch(
            'storage.user_store.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=MagicMock(current_org_id=test_org.id),
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.get_user_team_info',
            return_value={
                'spend': 25.50,
                'max_budget_in_team': 100.00,
            },
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.update_team_and_users_budget'
        ) as mock_update_budget,
    ):
        mock_stripe_retrieve.return_value = MagicMock(
            status='complete', amount_subtotal=2500, customer='mock_customer_id'
        )

        response = await success_callback(session_id, mock_request)

        assert response.status_code == 302
        assert (
            response.headers['location']
            == 'https://test.com/settings/billing?checkout=success'
        )

        mock_update_budget.assert_called_once_with(
            str(test_org.id),
            125.0,  # 100 + 25.00
        )

    # Verify database updates
    async with async_session_maker() as session:
        result = await session.execute(
            select(BillingSession).where(BillingSession.id == session_id)
        )
        billing_session = result.scalar_one_or_none()
        assert billing_session.status == 'completed'
        assert float(billing_session.price) == 25.0

        # Verify org byor_export_enabled was set
        org_result = await session.execute(select(Org).where(Org.id == test_org.id))
        org = org_result.scalar_one_or_none()
        assert org.byor_export_enabled is True


@pytest.mark.asyncio
async def test_success_callback_lite_llm_error(
    async_session_maker, test_org, test_user
):
    """Test handling of LiteLLM API errors during success callback."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    session_id = 'test_litellm_error_session'
    async with async_session_maker() as session:
        billing_session = BillingSession(
            id=session_id,
            user_id=str(test_user.id),
            org_id=test_org.id,
            status='in_progress',
            price=25,
            price_code='NA',
        )
        session.add(billing_session)
        await session.commit()

    with (
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
        patch(
            'storage.user_store.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=MagicMock(current_org_id=test_org.id),
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.get_user_team_info',
            side_effect=Exception('LiteLLM API Error'),
        ),
    ):
        mock_stripe_retrieve.return_value = MagicMock(
            status='complete', amount_subtotal=2500
        )

        with pytest.raises(Exception, match='LiteLLM API Error'):
            await success_callback(session_id, mock_request)

    # Verify no database updates occurred (transaction rolled back)
    async with async_session_maker() as session:
        result = await session.execute(
            select(BillingSession).where(BillingSession.id == session_id)
        )
        billing_session = result.scalar_one_or_none()
        assert billing_session.status == 'in_progress'


@pytest.mark.asyncio
async def test_success_callback_lite_llm_update_budget_error_rollback(
    async_session_maker, test_org, test_user
):
    """Test that database changes are not committed when update_team_and_users_budget fails.

    This test verifies that if LiteLlmManager.update_team_and_users_budget raises an exception,
    the database transaction rolls back.
    """
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    session_id = 'test_budget_rollback_session'
    async with async_session_maker() as session:
        billing_session = BillingSession(
            id=session_id,
            user_id=str(test_user.id),
            org_id=test_org.id,
            status='in_progress',
            price=10,
            price_code='NA',
        )
        session.add(billing_session)
        await session.commit()

    with (
        patch('server.routes.billing.a_session_maker', async_session_maker),
        patch('stripe.checkout.Session.retrieve') as mock_stripe_retrieve,
        patch(
            'storage.user_store.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=MagicMock(current_org_id=test_org.id),
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.get_user_team_info',
            return_value={
                'spend': 0,
                'max_budget_in_team': 0,
            },
        ),
        patch(
            'storage.lite_llm_manager.LiteLlmManager.update_team_and_users_budget',
            side_effect=Exception('LiteLLM API Error'),
        ),
    ):
        mock_stripe_retrieve.return_value = MagicMock(
            status='complete',
            amount_subtotal=1000,
            customer='mock_customer_id',
        )

        with pytest.raises(Exception, match='LiteLLM API Error'):
            await success_callback(session_id, mock_request)

    # Verify no database commit occurred - the transaction should roll back
    async with async_session_maker() as session:
        result = await session.execute(
            select(BillingSession).where(BillingSession.id == session_id)
        )
        billing_session = result.scalar_one_or_none()
        assert billing_session.status == 'in_progress'


@pytest.mark.asyncio
async def test_cancel_callback_session_not_found(async_session_maker):
    """Test cancel callback when billing session is not found."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    with patch('server.routes.billing.a_session_maker', async_session_maker):
        response = await cancel_callback('nonexistent_session_id', mock_request)
        assert response.status_code == 302
        assert (
            response.headers['location']
            == 'https://test.com/settings/billing?checkout=cancel'
        )


@pytest.mark.asyncio
async def test_cancel_callback_success(async_session_maker, test_org, test_user):
    """Test successful cancellation of billing session."""
    mock_request = Request(scope={'type': 'http'})
    mock_request._url = URL('http://test.com/')

    session_id = 'test_cancel_session'
    async with async_session_maker() as session:
        billing_session = BillingSession(
            id=session_id,
            user_id=str(test_user.id),
            org_id=test_org.id,
            status='in_progress',
            price=25,
            price_code='NA',
        )
        session.add(billing_session)
        await session.commit()

    with patch('server.routes.billing.a_session_maker', async_session_maker):
        response = await cancel_callback(session_id, mock_request)

        assert response.status_code == 302
        assert (
            response.headers['location']
            == 'https://test.com/settings/billing?checkout=cancel'
        )

    # Verify database update
    async with async_session_maker() as session:
        result = await session.execute(
            select(BillingSession).where(BillingSession.id == session_id)
        )
        billing_session = result.scalar_one_or_none()
        assert billing_session.status == 'cancelled'


@pytest.mark.asyncio
async def test_has_payment_method_with_payment_method():
    """Test has_payment_method returns True when user has a payment method."""
    mock_has_payment_method = AsyncMock(return_value=True)
    with patch(
        'server.routes.billing.stripe_service.has_payment_method_by_user_id',
        mock_has_payment_method,
    ):
        result = await has_payment_method('mock_user')
        assert result is True
    mock_has_payment_method.assert_called_once_with('mock_user')


@pytest.mark.asyncio
async def test_has_payment_method_without_payment_method():
    """Test has_payment_method returns False when user has no payment method."""
    mock_has_payment_method = AsyncMock(return_value=False)
    with patch(
        'server.routes.billing.stripe_service.has_payment_method_by_user_id',
        mock_has_payment_method,
    ):
        mock_has_payment_method.return_value = False
        result = await has_payment_method('mock_user')
        assert result is False
    mock_has_payment_method.assert_called_once_with('mock_user')


@pytest.mark.asyncio
async def test_create_customer_setup_session_success():
    """Test successful creation of customer setup session."""
    mock_request = Request(
        scope={
            'type': 'http',
            'path': '/api/billing/create-customer-setup-session',
            'server': ('test.com', 80),
            'headers': [],
        }
    )
    mock_request._url = URL('http://test.com/')

    mock_customer_info = {'customer_id': 'mock-customer-id', 'org_id': 'mock-org-id'}
    mock_session = MagicMock()
    mock_session.url = 'https://checkout.stripe.com/test-session'
    mock_create = AsyncMock(return_value=mock_session)

    with (
        patch(
            'integrations.stripe_service.find_or_create_customer_by_user_id',
            AsyncMock(return_value=mock_customer_info),
        ),
        patch('stripe.checkout.Session.create_async', mock_create),
        patch('server.routes.billing.validate_billing_enabled'),
    ):
        result = await create_customer_setup_session(mock_request, 'mock_user')

        assert isinstance(result, billing.CreateBillingSessionResponse)
        assert result.redirect_url == 'https://checkout.stripe.com/test-session'

        # Verify Stripe session creation parameters
        mock_create.assert_called_once_with(
            customer='mock-customer-id',
            mode='setup',
            payment_method_types=['card'],
            success_url='https://test.com?setup=success',
            cancel_url='https://test.com',
        )
