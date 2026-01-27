from uuid import UUID

import stripe
from server.constants import STRIPE_API_KEY
from server.logger import logger
from sqlalchemy.orm import Session
from storage.database import session_maker
from storage.org import Org
from storage.org_store import OrgStore
from storage.stripe_customer import StripeCustomer

from openhands.utils.async_utils import call_sync_from_async

stripe.api_key = STRIPE_API_KEY


async def find_customer_id_by_org_id(org_id: UUID) -> str | None:
    with session_maker() as session:
        stripe_customer = (
            session.query(StripeCustomer)
            .filter(StripeCustomer.org_id == org_id)
            .first()
        )
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


async def find_customer_id_by_user_id(user_id: str) -> str | None:
    # First search our own DB...
    org = await call_sync_from_async(
        OrgStore.get_current_org_from_keycloak_user_id, user_id
    )
    if not org:
        logger.warning(f'Org not found for user {user_id}')
        return None
    customer_id = await find_customer_id_by_org_id(org.id)
    return customer_id


async def find_or_create_customer_by_user_id(user_id: str) -> dict | None:
    # Get the current org for the user
    org = await call_sync_from_async(
        OrgStore.get_current_org_from_keycloak_user_id, user_id
    )
    if not org:
        logger.warning(f'Org not found for user {user_id}')
        return None

    customer_id = await find_customer_id_by_org_id(org.id)
    if customer_id:
        return {'customer_id': customer_id, 'org_id': str(org.id)}
    logger.info(
        'creating_customer',
        extra={'user_id': user_id, 'org_id': str(org.id)},
    )

    # Create the customer in stripe
    customer = await stripe.Customer.create_async(
        email=org.contact_email,
        metadata={'org_id': str(org.id)},
    )

    # Save the stripe customer in the local db
    with session_maker() as session:
        session.add(
            StripeCustomer(
                keycloak_user_id=user_id,
                org_id=org.id,
                stripe_customer_id=customer.id,
            )
        )
        session.commit()

    logger.info(
        'created_customer',
        extra={
            'user_id': user_id,
            'org_id': str(org.id),
            'stripe_customer_id': customer.id,
        },
    )
    return {'customer_id': customer.id, 'org_id': str(org.id)}


async def has_payment_method_by_user_id(user_id: str) -> bool:
    customer_id = await find_customer_id_by_user_id(user_id)
    if customer_id is None:
        return False
    payment_methods = await stripe.Customer.list_payment_methods_async(
        customer_id,
    )
    logger.info(
        f'has_payment_method:{user_id}:{customer_id}:{bool(payment_methods.data)}'
    )
    return bool(payment_methods.data)


async def migrate_customer(session: Session, user_id: str, org: Org):
    stripe_customer = (
        session.query(StripeCustomer)
        .filter(StripeCustomer.keycloak_user_id == user_id)
        .first()
    )
    if stripe_customer is None:
        return
    stripe_customer.org_id = org.id
    customer = await stripe.Customer.modify_async(
        id=stripe_customer.stripe_customer_id,
        email=org.contact_email,
        metadata={'user_id': '', 'org_id': str(org.id)},
    )

    logger.info(
        'migrated_customer',
        extra={
            'user_id': user_id,
            'org_id': str(org.id),
            'stripe_customer_id': customer.id,
        },
    )
