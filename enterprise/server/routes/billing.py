# billing.py - Handles all billing-related operations including credit management and Stripe integration
import typing
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import httpx
import stripe
from dateutil.relativedelta import relativedelta  # type: ignore
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from integrations import stripe_service
from pydantic import BaseModel
from server.config import get_config
from server.constants import (
    LITE_LLM_API_KEY,
    LITE_LLM_API_URL,
    STRIPE_API_KEY,
    STRIPE_WEBHOOK_SECRET,
    SUBSCRIPTION_PRICE_DATA,
    get_default_litellm_model,
)
from server.logger import logger
from storage.billing_session import BillingSession
from storage.database import session_maker
from storage.saas_settings_store import SaasSettingsStore
from storage.subscription_access import SubscriptionAccess

from openhands.server.user_auth import get_user_id
from openhands.utils.http_session import httpx_verify_option

stripe.api_key = STRIPE_API_KEY
billing_router = APIRouter(prefix='/api/billing')


# TODO: Add a new app_mode named "ON_PREM" to support self-hosted customers instead of doing this
# and members should comment out the "validate_saas_environment" function if they are developing and testing locally.
def is_all_hands_saas_environment(request: Request) -> bool:
    """Check if the current domain is an All Hands SaaS environment.

    Args:
        request: FastAPI Request object

    Returns:
        True if the current domain contains "all-hands.dev" or "openhands.dev" postfix
    """
    hostname = request.url.hostname or ''
    return hostname.endswith('all-hands.dev') or hostname.endswith('openhands.dev')


def validate_saas_environment(request: Request) -> None:
    """Validate that the request is coming from an All Hands SaaS environment.

    Args:
        request: FastAPI Request object

    Raises:
        HTTPException: If the request is not from an All Hands SaaS environment
    """
    if not is_all_hands_saas_environment(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Checkout sessions are only available for All Hands SaaS environments',
        )


class BillingSessionType(Enum):
    DIRECT_PAYMENT = 'DIRECT_PAYMENT'
    MONTHLY_SUBSCRIPTION = 'MONTHLY_SUBSCRIPTION'


class GetCreditsResponse(BaseModel):
    credits: Decimal | None = None


class SubscriptionAccessResponse(BaseModel):
    start_at: datetime
    end_at: datetime
    created_at: datetime
    cancelled_at: datetime | None = None
    stripe_subscription_id: str | None = None


class CreateCheckoutSessionRequest(BaseModel):
    amount: int


class CreateBillingSessionResponse(BaseModel):
    redirect_url: str


class GetSessionStatusResponse(BaseModel):
    status: str
    customer_email: str


class LiteLlmUserInfo(typing.TypedDict, total=False):
    max_budget: float | None
    spend: float | None


def calculate_credits(user_info: LiteLlmUserInfo) -> float:
    # using `or` after get with default because it could be missing or present as None.
    max_budget = user_info.get('max_budget') or 0.0
    spend = user_info.get('spend') or 0.0
    return max(max_budget - spend, 0.0)


# Endpoint to retrieve user's current credit balance
@billing_router.get('/credits')
async def get_credits(user_id: str = Depends(get_user_id)) -> GetCreditsResponse:
    if not stripe_service.STRIPE_API_KEY:
        return GetCreditsResponse()
    async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
        user_json = await _get_litellm_user(client, user_id)
        credits = calculate_credits(user_json['user_info'])
    return GetCreditsResponse(credits=Decimal('{:.2f}'.format(credits)))


# Endpoint to retrieve user's current subscription access
@billing_router.get('/subscription-access')
async def get_subscription_access(
    user_id: str = Depends(get_user_id),
) -> SubscriptionAccessResponse | None:
    """Get details of the currently valid subscription for the user."""
    with session_maker() as session:
        now = datetime.now(UTC)
        subscription_access = (
            session.query(SubscriptionAccess)
            .filter(SubscriptionAccess.status == 'ACTIVE')
            .filter(SubscriptionAccess.user_id == user_id)
            .filter(SubscriptionAccess.start_at <= now)
            .filter(SubscriptionAccess.end_at >= now)
            .first()
        )
        if not subscription_access:
            return None
        return SubscriptionAccessResponse(
            start_at=subscription_access.start_at,
            end_at=subscription_access.end_at,
            created_at=subscription_access.created_at,
            cancelled_at=subscription_access.cancelled_at,
            stripe_subscription_id=subscription_access.stripe_subscription_id,
        )


# Endpoint to check if a user has entered a payment method into stripe
@billing_router.post('/has-payment-method')
async def has_payment_method(user_id: str = Depends(get_user_id)) -> bool:
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    return await stripe_service.has_payment_method(user_id)


# Endpoint to cancel user's subscription
@billing_router.post('/cancel-subscription')
async def cancel_subscription(user_id: str = Depends(get_user_id)) -> JSONResponse:
    """Cancel user's active subscription at the end of the current billing period."""
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)

    with session_maker() as session:
        # Find the user's active subscription
        now = datetime.now(UTC)
        subscription_access = (
            session.query(SubscriptionAccess)
            .filter(SubscriptionAccess.status == 'ACTIVE')
            .filter(SubscriptionAccess.user_id == user_id)
            .filter(SubscriptionAccess.start_at <= now)
            .filter(SubscriptionAccess.end_at >= now)
            .filter(SubscriptionAccess.cancelled_at.is_(None))  # Not already cancelled
            .first()
        )

        if not subscription_access:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='No active subscription found',
            )

        if not subscription_access.stripe_subscription_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Cannot cancel subscription: missing Stripe subscription ID',
            )

        try:
            # Cancel the subscription in Stripe at period end
            await stripe.Subscription.modify_async(
                subscription_access.stripe_subscription_id, cancel_at_period_end=True
            )

            # Update local database
            subscription_access.cancelled_at = datetime.now(UTC)
            session.merge(subscription_access)
            session.commit()

            logger.info(
                'subscription_cancelled',
                extra={
                    'user_id': user_id,
                    'stripe_subscription_id': subscription_access.stripe_subscription_id,
                    'subscription_access_id': subscription_access.id,
                    'end_at': subscription_access.end_at,
                },
            )

            return JSONResponse(
                {'status': 'success', 'message': 'Subscription cancelled successfully'}
            )

        except stripe.StripeError as e:
            logger.error(
                'stripe_cancellation_failed',
                extra={
                    'user_id': user_id,
                    'stripe_subscription_id': subscription_access.stripe_subscription_id,
                    'error': str(e),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f'Failed to cancel subscription: {str(e)}',
            )


# Endpoint to create a new setup intent in stripe
@billing_router.post('/create-customer-setup-session')
async def create_customer_setup_session(
    request: Request, user_id: str = Depends(get_user_id)
) -> CreateBillingSessionResponse:
    validate_saas_environment(request)

    customer_id = await stripe_service.find_or_create_customer(user_id)
    checkout_session = await stripe.checkout.Session.create_async(
        customer=customer_id,
        mode='setup',
        payment_method_types=['card'],
        success_url=f'{request.base_url}?free_credits=success',
        cancel_url=f'{request.base_url}',
    )
    return CreateBillingSessionResponse(redirect_url=checkout_session.url)  # type: ignore[arg-type]


# Endpoint to create a new Stripe checkout session for credit purchase
@billing_router.post('/create-checkout-session')
async def create_checkout_session(
    body: CreateCheckoutSessionRequest,
    request: Request,
    user_id: str = Depends(get_user_id),
) -> CreateBillingSessionResponse:
    validate_saas_environment(request)

    customer_id = await stripe_service.find_or_create_customer(user_id)
    checkout_session = await stripe.checkout.Session.create_async(
        customer=customer_id,
        line_items=[
            {
                'price_data': {
                    'unit_amount': body.amount * 100,
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
        saved_payment_method_options={
            'payment_method_save': 'enabled',
        },
        success_url=f'{request.base_url}api/billing/success?session_id={{CHECKOUT_SESSION_ID}}',
        cancel_url=f'{request.base_url}api/billing/cancel?session_id={{CHECKOUT_SESSION_ID}}',
    )
    logger.info(
        'created_stripe_checkout_session',
        extra={
            'stripe_customer_id': customer_id,
            'user_id': user_id,
            'amount': body.amount,
            'checkout_session_id': checkout_session.id,
        },
    )
    with session_maker() as session:
        billing_session = BillingSession(
            id=checkout_session.id,
            user_id=user_id,
            price=body.amount,
            price_code='NA',
            billing_session_type=BillingSessionType.DIRECT_PAYMENT.value,
        )
        session.add(billing_session)
        session.commit()

    return CreateBillingSessionResponse(redirect_url=checkout_session.url)  # type: ignore[arg-type]


@billing_router.post('/subscription-checkout-session')
async def create_subscription_checkout_session(
    request: Request,
    billing_session_type: BillingSessionType = BillingSessionType.MONTHLY_SUBSCRIPTION,
    user_id: str = Depends(get_user_id),
) -> CreateBillingSessionResponse:
    validate_saas_environment(request)

    # Prevent duplicate subscriptions for the same user
    with session_maker() as session:
        now = datetime.now(UTC)
        existing_active_subscription = (
            session.query(SubscriptionAccess)
            .filter(SubscriptionAccess.status == 'ACTIVE')
            .filter(SubscriptionAccess.user_id == user_id)
            .filter(SubscriptionAccess.start_at <= now)
            .filter(SubscriptionAccess.end_at >= now)
            .filter(SubscriptionAccess.cancelled_at.is_(None))  # Not cancelled
            .first()
        )

        if existing_active_subscription:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Cannot create subscription: User already has an active subscription that has not been cancelled',
            )

    customer_id = await stripe_service.find_or_create_customer(user_id)
    subscription_price_data = SUBSCRIPTION_PRICE_DATA[billing_session_type.value]
    checkout_session = await stripe.checkout.Session.create_async(
        customer=customer_id,
        line_items=[
            {
                'price_data': subscription_price_data,
                'quantity': 1,
            }
        ],
        mode='subscription',
        payment_method_types=['card'],
        saved_payment_method_options={
            'payment_method_save': 'enabled',
        },
        success_url=f'{request.base_url}api/billing/success?session_id={{CHECKOUT_SESSION_ID}}',
        cancel_url=f'{request.base_url}api/billing/cancel?session_id={{CHECKOUT_SESSION_ID}}',
        subscription_data={
            'metadata': {
                'user_id': user_id,
                'billing_session_type': billing_session_type.value,
            }
        },
    )
    logger.info(
        'created_stripe_subscription_checkout_session',
        extra={
            'stripe_customer_id': customer_id,
            'user_id': user_id,
            'checkout_session_id': checkout_session.id,
            'billing_session_type': billing_session_type.value,
        },
    )
    with session_maker() as session:
        billing_session = BillingSession(
            id=checkout_session.id,
            user_id=user_id,
            price=subscription_price_data['unit_amount'],
            price_code='NA',
            billing_session_type=billing_session_type.value,
        )
        session.add(billing_session)
        session.commit()

    return CreateBillingSessionResponse(
        redirect_url=typing.cast(str, checkout_session.url)
    )


@billing_router.get('/create-subscription-checkout-session')
async def create_subscription_checkout_session_via_get(
    request: Request,
    billing_session_type: BillingSessionType = BillingSessionType.MONTHLY_SUBSCRIPTION,
    user_id: str = Depends(get_user_id),
) -> RedirectResponse:
    """Create a subscription checkout session using a GET request (For easier copy / paste to URL bar)."""
    validate_saas_environment(request)

    response = await create_subscription_checkout_session(
        request, billing_session_type, user_id
    )
    return RedirectResponse(response.redirect_url)


# Callback endpoint for successful Stripe payments - updates user credits and billing session status
@billing_router.get('/success')
async def success_callback(session_id: str, request: Request):
    # We can't use the auth cookie because of SameSite=strict
    with session_maker() as session:
        billing_session = (
            session.query(BillingSession)
            .filter(BillingSession.id == session_id)
            .filter(BillingSession.status == 'in_progress')
            .first()
        )

        if billing_session is None:
            # Hopefully this never happens - we get a redirect from stripe where the session does not exist
            logger.error(
                'session_id_not_found', extra={'checkout_session_id': session_id}
            )
            raise HTTPException(status.HTTP_400_BAD_REQUEST)

        # Any non direct payment (Subscription) is processed in the invoice_payment.paid by the webhook
        if (
            billing_session.billing_session_type
            != BillingSessionType.DIRECT_PAYMENT.value
        ):
            return RedirectResponse(
                f'{request.base_url}settings?checkout=success', status_code=302
            )

        stripe_session = stripe.checkout.Session.retrieve(session_id)
        if stripe_session.status != 'complete':
            # Hopefully this never happens - we get a redirect from stripe where the payment is not yet complete
            # (Or somebody tried to manually build the URL)
            logger.error(
                'payment_not_complete',
                extra={
                    'checkout_session_id': session_id,
                    'stripe_customer_id': stripe_session.customer,
                },
            )
            raise HTTPException(status.HTTP_400_BAD_REQUEST)

        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            # Update max budget in litellm
            user_json = await _get_litellm_user(client, billing_session.user_id)
            amount_subtotal = stripe_session.amount_subtotal or 0
            add_credits = amount_subtotal / 100
            new_max_budget = (
                (user_json.get('user_info') or {}).get('max_budget') or 0
            ) + add_credits
            await _upsert_litellm_user(client, billing_session.user_id, new_max_budget)

            # Store transaction status
            billing_session.status = 'completed'
            billing_session.price = amount_subtotal
            billing_session.updated_at = datetime.now(UTC)
            session.merge(billing_session)
            logger.info(
                'stripe_checkout_success',
                extra={
                    'amount_subtotal': stripe_session.amount_subtotal,
                    'user_id': billing_session.user_id,
                    'checkout_session_id': billing_session.id,
                    'stripe_customer_id': stripe_session.customer,
                },
            )
            session.commit()

    return RedirectResponse(
        f'{request.base_url}settings/billing?checkout=success', status_code=302
    )


# Callback endpoint for cancelled Stripe payments - updates billing session status
@billing_router.get('/cancel')
async def cancel_callback(session_id: str, request: Request):
    with session_maker() as session:
        billing_session = (
            session.query(BillingSession)
            .filter(BillingSession.id == session_id)
            .filter(BillingSession.status == 'in_progress')
            .first()
        )
        if billing_session:
            logger.info(
                'stripe_checkout_cancel',
                extra={
                    'user_id': billing_session.user_id,
                    'checkout_session_id': billing_session.id,
                },
            )
            billing_session.status = 'cancelled'
            billing_session.updated_at = datetime.now(UTC)
            session.merge(billing_session)
            session.commit()

            # Redirect credit purchases to billing screen, subscriptions to LLM settings
            if (
                billing_session.billing_session_type
                == BillingSessionType.DIRECT_PAYMENT.value
            ):
                return RedirectResponse(
                    f'{request.base_url}settings/billing?checkout=cancel',
                    status_code=302,
                )
            else:
                return RedirectResponse(
                    f'{request.base_url}settings?checkout=cancel', status_code=302
                )

    # If no billing session found, default to LLM settings (subscription flow)
    return RedirectResponse(
        f'{request.base_url}settings?checkout=cancel', status_code=302
    )


@billing_router.post('/stripe-webhook')
async def stripe_webhook(request: Request) -> JSONResponse:
    """Endpoint for stripe webhooks."""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        # Invalid payload
        raise HTTPException(status_code=400, detail=f'Invalid payload: {e}')
    except stripe.SignatureVerificationError as e:
        # Invalid signature
        raise HTTPException(status_code=400, detail=f'Invalid signature: {e}')

    # Handle the event
    logger.info('stripe_webhook_event', extra={'event': event})
    event_type = event['type']
    if event_type == 'invoice.paid':
        invoice = event['data']['object']
        amount_paid = invoice.amount_paid
        metadata = invoice.parent.subscription_details.metadata  # type: ignore
        billing_session_type = metadata.billing_session_type
        assert (
            amount_paid == SUBSCRIPTION_PRICE_DATA[billing_session_type]['unit_amount']
        )
        user_id = metadata.user_id

        start_at = datetime.now(UTC)
        if billing_session_type == BillingSessionType.MONTHLY_SUBSCRIPTION.value:
            end_at = start_at + relativedelta(months=1)
        else:
            raise ValueError(f'unknown_billing_session_type:{billing_session_type}')

        with session_maker() as session:
            subscription_access = SubscriptionAccess(
                status='ACTIVE',
                user_id=user_id,
                start_at=start_at,
                end_at=end_at,
                amount_paid=amount_paid,
                stripe_invoice_payment_id=invoice.payment_intent,
                stripe_subscription_id=invoice.subscription,  # Store Stripe subscription ID
            )
            session.add(subscription_access)
            session.commit()
    elif event_type == 'customer.subscription.updated':
        subscription = event['data']['object']
        subscription_id = subscription['id']

        # Handle subscription cancellation
        if subscription.get('cancel_at_period_end') is True:
            with session_maker() as session:
                subscription_access = (
                    session.query(SubscriptionAccess)
                    .filter(
                        SubscriptionAccess.stripe_subscription_id == subscription_id
                    )
                    .filter(SubscriptionAccess.status == 'ACTIVE')
                    .first()
                )

                if subscription_access and not subscription_access.cancelled_at:
                    subscription_access.cancelled_at = datetime.now(UTC)
                    session.merge(subscription_access)
                    session.commit()

                    logger.info(
                        'subscription_cancelled_via_webhook',
                        extra={
                            'stripe_subscription_id': subscription_id,
                            'user_id': subscription_access.user_id,
                            'subscription_access_id': subscription_access.id,
                        },
                    )
    elif event_type == 'customer.subscription.deleted':
        subscription = event['data']['object']
        subscription_id = subscription['id']

        with session_maker() as session:
            subscription_access = (
                session.query(SubscriptionAccess)
                .filter(SubscriptionAccess.stripe_subscription_id == subscription_id)
                .filter(SubscriptionAccess.status == 'ACTIVE')
                .first()
            )

            if subscription_access:
                subscription_access.status = 'DISABLED'
                subscription_access.updated_at = datetime.now(UTC)
                session.merge(subscription_access)
                session.commit()

                # Reset user settings to free tier defaults
                reset_user_to_free_tier_settings(subscription_access.user_id)

                logger.info(
                    'subscription_expired_reset_to_free_tier',
                    extra={
                        'stripe_subscription_id': subscription_id,
                        'user_id': subscription_access.user_id,
                        'subscription_access_id': subscription_access.id,
                    },
                )
    else:
        logger.info('stripe_webhook_unhandled_event_type', extra={'type': event_type})

    return JSONResponse({'status': 'success'})


def reset_user_to_free_tier_settings(user_id: str) -> None:
    """Reset user settings to free tier defaults when subscription ends."""
    config = get_config()
    settings_store = SaasSettingsStore(
        user_id=user_id, session_maker=session_maker, config=config
    )

    with session_maker() as session:
        user_settings = settings_store.get_user_settings_by_keycloak_id(
            user_id, session
        )

        if user_settings:
            user_settings.llm_model = get_default_litellm_model()
            user_settings.llm_api_key = None
            user_settings.llm_api_key_for_byor = None
            user_settings.llm_base_url = LITE_LLM_API_URL
            user_settings.max_budget_per_task = None
            user_settings.confirmation_mode = False
            user_settings.enable_solvability_analysis = False
            user_settings.security_analyzer = 'llm'
            user_settings.agent = 'CodeActAgent'
            user_settings.language = 'en'
            user_settings.enable_default_condenser = True
            user_settings.enable_sound_notifications = False
            user_settings.enable_proactive_conversation_starters = True
            user_settings.user_consents_to_analytics = False

            session.merge(user_settings)
            session.commit()

            logger.info(
                'user_settings_reset_to_free_tier',
                extra={
                    'user_id': user_id,
                    'reset_timestamp': datetime.now(UTC).isoformat(),
                },
            )


async def _get_litellm_user(client: httpx.AsyncClient, user_id: str) -> dict:
    """Get a user from litellm with the id matching that given.

    If no such user exists, returns a dummy user in the format:
    `{'user_id': '<USER_ID>', 'user_info': {'spend': 0}, 'keys': [], 'teams': []}`
    """
    response = await client.get(
        f'{LITE_LLM_API_URL}/user/info?user_id={user_id}',
        headers={
            'x-goog-api-key': LITE_LLM_API_KEY,
        },
    )
    response.raise_for_status()
    return response.json()


async def _upsert_litellm_user(
    client: httpx.AsyncClient, user_id: str, max_budget: float
):
    """Insert / Update a user in litellm."""
    response = await client.post(
        f'{LITE_LLM_API_URL}/user/update',
        headers={
            'x-goog-api-key': LITE_LLM_API_KEY,
        },
        json={
            'user_id': user_id,
            'max_budget': max_budget,
        },
    )
    response.raise_for_status()
