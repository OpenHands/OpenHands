from uuid import UUID

import stripe
from server.constants import STRIPE_API_KEY
from server.logger import logger
from sqlalchemy import select
from storage.database import a_session_maker
from storage.org import Org
from storage.org_store import OrgStore
from storage.stripe_customer import StripeCustomer

stripe.api_key = STRIPE_API_KEY


async def find_customer_id_by_org_id(org_id: UUID) -> str | None:
    async with a_session_maker() as session:
        stmt = select(StripeCustomer).where(StripeCustomer.org_id == org_id)
        result = await session.execute(stmt)
        stripe_customer = result.scalar_one_or_none()
        if stripe_customer:
            return stripe_customer.stripe_customer_id

    # If that fails, fallback to stripe
    search_result = await stripe.Customer.search_async(
        query=f"metadata['org_id']:'{str(org_id)}'",
    )
    data = search_result.data
    if not data:
        logger.info(
            'no_customer_for_org_id',
            extra={'org_id': str(org_id)},
        )
        return None
    return data[0].id  # type: ignore [attr-defined]


async def _resolve_org_for_user(user_id: str, org_id: UUID | None) -> Org | None:
    """Resolve an explicit ``org_id`` to an Org, or fall back to the
    user's current org. Returns ``None`` if neither resolves.
    """
    if org_id is not None:
        org = await OrgStore.get_org_by_id(org_id)
        if not org:
            logger.warning(
                'stripe_org_not_found_for_id',
                extra={'user_id': user_id, 'org_id': str(org_id)},
            )
        return org
    org = await OrgStore.get_current_org_from_keycloak_user_id(user_id)
    if not org:
        logger.warning(f'Org not found for user {user_id}')
    return org


async def find_customer_id_by_user_id(
    user_id: str, org_id: UUID | None = None
) -> str | None:
    """Look up the Stripe customer id for the user's billing org.

    ``org_id`` should be the request's effective org (X-Org-Id / API-key
    binding). When ``None``, falls back to the user's persisted current
    org for backwards compatibility with non-request contexts.
    """
    org = await _resolve_org_for_user(user_id, org_id)
    if not org:
        return None
    return await find_customer_id_by_org_id(org.id)


async def find_or_create_customer_by_user_id(
    user_id: str, org_id: UUID | None = None
) -> dict | None:
    """Find or create the Stripe customer for the user's billing org.

    See :func:`find_customer_id_by_user_id` for the ``org_id`` semantics.
    """
    org = await _resolve_org_for_user(user_id, org_id)
    if not org:
        return None

    customer_id = await find_customer_id_by_org_id(org.id)
    if customer_id:
        return {'customer_id': customer_id, 'org_id': str(org.id)}
    logger.info(
        'creating_customer',
        extra={'user_id': user_id, 'org_id': str(org.id)},
    )

    # Create the customer in stripe (only include email if available)
    create_params: dict = {'metadata': {'org_id': str(org.id)}}
    if org.contact_email:
        create_params['email'] = org.contact_email
    customer = await stripe.Customer.create_async(**create_params)

    # Save the stripe customer in the local db
    async with a_session_maker() as session:
        session.add(
            StripeCustomer(
                keycloak_user_id=user_id,
                org_id=org.id,
                stripe_customer_id=customer.id,
            )
        )
        await session.commit()

    logger.info(
        'created_customer',
        extra={
            'user_id': user_id,
            'org_id': str(org.id),
            'stripe_customer_id': customer.id,
        },
    )
    return {'customer_id': customer.id, 'org_id': str(org.id)}


async def has_payment_method_by_user_id(
    user_id: str, org_id: UUID | None = None
) -> bool:
    customer_id = await find_customer_id_by_user_id(user_id, org_id=org_id)
    if customer_id is None:
        return False
    payment_methods = await stripe.Customer.list_payment_methods_async(
        customer_id,
    )
    logger.info(
        f'has_payment_method:{user_id}:{customer_id}:{bool(payment_methods.data)}'
    )
    return bool(payment_methods.data)


async def migrate_customer(session, user_id: str, org: Org):
    result = await session.execute(
        select(StripeCustomer).where(StripeCustomer.keycloak_user_id == user_id)
    )
    stripe_customer = result.scalar_one_or_none()
    if stripe_customer is None:
        return
    stripe_customer.org_id = org.id
    # Only include email if available to avoid sending empty strings to Stripe
    modify_params: dict = {
        'id': stripe_customer.stripe_customer_id,
        'metadata': {'user_id': '', 'org_id': str(org.id)},
    }
    if org.contact_email:
        modify_params['email'] = org.contact_email
    customer = await stripe.Customer.modify_async(**modify_params)

    logger.info(
        'migrated_customer',
        extra={
            'user_id': user_id,
            'org_id': str(org.id),
            'stripe_customer_id': customer.id,
        },
    )
