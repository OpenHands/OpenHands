import stripe
from server.auth.token_manager import TokenManager
from server.constants import STRIPE_API_KEY
from server.logger import logger
from storage.database import session_maker
from storage.stripe_customer import StripeCustomer

stripe.api_key = STRIPE_API_KEY


async def _validate_and_cleanup_customer_id(user_id: str, customer_id: str) -> str | None:
    """
    Validate that a customer ID exists in Stripe. If not, remove it from local database.
    Returns the customer_id if valid, None if invalid.
    If no Stripe API key is configured, returns the customer_id without validation.
    """
    # Skip validation if no API key is configured (e.g., in tests)
    if not STRIPE_API_KEY:
        logger.debug(
            'skipping_customer_validation_no_api_key',
            extra={'user_id': user_id, 'customer_id': customer_id},
        )
        return customer_id
    
    try:
        # Try to retrieve the customer from Stripe to validate it exists
        await stripe.Customer.retrieve_async(customer_id)
        return customer_id
    except stripe.AuthenticationError:
        # If authentication fails, skip validation (e.g., in tests)
        logger.debug(
            'skipping_customer_validation_auth_error',
            extra={'user_id': user_id, 'customer_id': customer_id},
        )
        return customer_id
    except stripe.InvalidRequestError as e:
        if 'No such customer' in str(e):
            logger.warning(
                'stale_customer_id_found_removing_from_db',
                extra={
                    'user_id': user_id,
                    'stale_customer_id': customer_id,
                    'error': str(e),
                },
            )
            # Remove the stale customer ID from local database
            with session_maker() as session:
                stripe_customer = (
                    session.query(StripeCustomer)
                    .filter(StripeCustomer.keycloak_user_id == user_id)
                    .filter(StripeCustomer.stripe_customer_id == customer_id)
                    .first()
                )
                if stripe_customer:
                    session.delete(stripe_customer)
                    session.commit()
                    logger.info(
                        'removed_stale_customer_id_from_db',
                        extra={
                            'user_id': user_id,
                            'removed_customer_id': customer_id,
                        },
                    )
            return None
        else:
            # Re-raise other Stripe errors
            raise


async def find_customer_id_by_user_id(user_id: str) -> str | None:
    # First search our own DB...
    with session_maker() as session:
        stripe_customer = (
            session.query(StripeCustomer)
            .filter(StripeCustomer.keycloak_user_id == user_id)
            .first()
        )
        if stripe_customer:
            # Validate that the stored customer ID still exists in Stripe
            validated_customer_id = await _validate_and_cleanup_customer_id(
                user_id, stripe_customer.stripe_customer_id
            )
            if validated_customer_id:
                return validated_customer_id
            # If validation failed, the stale ID was removed, continue to fallback

    # If that fails, fallback to stripe
    search_result = await stripe.Customer.search_async(
        query=f"metadata['user_id']:'{user_id}'",
    )
    data = search_result.data
    if not data:
        logger.info('no_customer_for_user_id', extra={'user_id': user_id})
        return None
    return data[0].id  # type: ignore [attr-defined]


async def find_or_create_customer(user_id: str) -> str:
    customer_id = await find_customer_id_by_user_id(user_id)
    if customer_id:
        return customer_id
    logger.info('creating_customer', extra={'user_id': user_id})

    # Get the user info from keycloak
    token_manager = TokenManager()
    user_info = await token_manager.get_user_info_from_user_id(user_id) or {}

    # Create the customer in stripe
    customer = await stripe.Customer.create_async(
        email=str(user_info.get('email', '')),
        metadata={'user_id': user_id},
    )

    # Save the stripe customer in the local db
    with session_maker() as session:
        session.add(
            StripeCustomer(keycloak_user_id=user_id, stripe_customer_id=customer.id)
        )
        session.commit()

    logger.info(
        'created_customer',
        extra={'user_id': user_id, 'stripe_customer_id': customer.id},
    )
    return customer.id


async def has_payment_method(user_id: str) -> bool:
    customer_id = await find_customer_id_by_user_id(user_id)
    if customer_id is None:
        return False
    
    try:
        payment_methods = await stripe.Customer.list_payment_methods_async(
            customer_id,
        )
        logger.info(
            f'has_payment_method:{user_id}:{customer_id}:{bool(payment_methods.data)}'
        )
        return bool(payment_methods.data)
    except stripe.InvalidRequestError as e:
        if 'No such customer' in str(e):
            logger.warning(
                'customer_not_found_in_has_payment_method',
                extra={
                    'user_id': user_id,
                    'customer_id': customer_id,
                    'error': str(e),
                },
            )
            # The customer was already validated in find_customer_id_by_user_id,
            # but if it still fails here, return False
            return False
        else:
            # Re-raise other Stripe errors
            raise
