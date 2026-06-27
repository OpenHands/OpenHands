#!/usr/bin/env python3
"""Sync OpenHands users to a Resend.com audience.

This script reads users from the OpenHands application database and adds them to
a Resend.com audience. It handles rate limiting and retries with exponential
backoff for adding contacts. When a user is newly added to the mailing list, a
welcome email is sent.

Required environment variables:
- RESEND_API_KEY: API key for Resend.com
- RESEND_AUDIENCE_ID: ID of the Resend audience to add users to

Optional environment variables:
- RESEND_FROM_EMAIL: Email address to use as the sender
  (default: "OpenHands Team <no-reply@welcome.openhands.dev>")
- RESEND_REPLY_TO_EMAIL: Email address for replies (default: "contact@openhands.dev")
- BATCH_SIZE: Number of users to process in each batch (default: 2000)
- MAX_RETRIES: Maximum number of retries for API calls (default: 3)
- INITIAL_BACKOFF_SECONDS: Initial backoff time for retries (default: 1)
- MAX_BACKOFF_SECONDS: Maximum backoff time for retries (default: 60)
- BACKOFF_FACTOR: Backoff factor for retries (default: 2)
- RATE_LIMIT: Rate limit for API calls (requests per second) (default: 2)
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import resend
from resend.exceptions import ResendError
from sqlalchemy import and_, func, select
from storage.resend_synced_user_store import ResendSyncedUserStore
from storage.user import User
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from openhands.app_server.utils.logger import openhands_logger as logger

# Get configuration from environment variables
RESEND_API_KEY = os.environ.get('RESEND_API_KEY')
RESEND_AUDIENCE_ID = os.environ.get('RESEND_AUDIENCE_ID', '')

# Sync configuration
# BATCH_SIZE controls only local DB pagination. Resend API calls remain
# individually rate-limited by RATE_LIMIT, so increasing this avoids frequent
# DB count/page queries without creating a 2000-contact API burst.
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '2000'))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
INITIAL_BACKOFF_SECONDS = float(os.environ.get('INITIAL_BACKOFF_SECONDS', '1'))
MAX_BACKOFF_SECONDS = float(os.environ.get('MAX_BACKOFF_SECONDS', '60'))
BACKOFF_FACTOR = float(os.environ.get('BACKOFF_FACTOR', '2'))
RATE_LIMIT = float(os.environ.get('RATE_LIMIT', '2'))  # Requests per second

# Set up Resend API
resend.api_key = RESEND_API_KEY


@dataclass(frozen=True)
class ResendUser:
    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None


class ResendSyncError(Exception):
    """Base exception for Resend sync errors."""

    pass


class ResendAPIError(ResendSyncError):
    """Exception for Resend API errors."""

    pass


# Email validation regex pattern - matches standard email format
# This pattern is intentionally strict to avoid Resend API validation errors
# It rejects special characters like ! that some email providers technically allow
# but Resend's API does not accept
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def _get_session_maker():
    """Get the application database session maker."""
    from openhands.app_server.config import get_global_config

    config = get_global_config()
    return config.db_session.get_session_maker()


def _valid_user_email_filter():
    return and_(User.email.isnot(None), User.email != '')


def _split_display_name(display_name: str | None) -> tuple[str | None, str | None]:
    """Split a stored display name into Resend first/last name fields."""
    if not display_name or not display_name.strip():
        return None, None

    parts = display_name.strip().split(maxsplit=1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else None
    return first_name, last_name


def get_local_users(offset: int = 0, limit: int = BATCH_SIZE) -> list[ResendUser]:
    """Get users with email addresses from the OpenHands database."""
    session_maker = _get_session_maker()
    with session_maker() as session:
        rows = session.execute(
            select(User.id, User.email, User.git_user_name)
            .where(_valid_user_email_filter())
            .order_by(User.id)
            .offset(offset)
            .limit(limit)
        ).all()

    users = []
    for row in rows:
        first_name, last_name = _split_display_name(row.git_user_name)
        users.append(
            ResendUser(
                id=str(row.id),
                email=row.email,
                first_name=first_name,
                last_name=last_name,
            )
        )
    return users


def get_total_local_users() -> int:
    """Get the total number of OpenHands users with email addresses."""
    session_maker = _get_session_maker()
    with session_maker() as session:
        return (
            session.execute(
                select(func.count()).select_from(User).where(_valid_user_email_filter())
            ).scalar()
            or 0
        )


def is_valid_email(email: Optional[str]) -> bool:
    """Validate an email address format.

    This uses a regex pattern that matches most valid email addresses
    while rejecting addresses with special characters that Resend's API
    does not accept (e.g., exclamation marks).

    Args:
        email: The email address to validate, or None.

    Returns:
        True if the email is valid, False otherwise (including for None).
    """
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email))


def get_resend_contacts(audience_id: str) -> Dict[str, Dict[str, Any]]:
    """Get contacts from Resend.

    Args:
        audience_id: The Resend audience ID.

    Returns:
        A dictionary mapping email addresses to contact data.

    Raises:
        ResendAPIError: If the API call fails.
    """
    try:
        contacts = resend.Contacts.list(audience_id).get('data', [])
        # Create a dictionary mapping email addresses to contact data for
        # efficient lookup
        return {contact['email'].lower(): contact for contact in contacts}
    except Exception:
        logger.exception('Failed to get contacts from Resend')
        raise


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(
        multiplier=INITIAL_BACKOFF_SECONDS,
        max=MAX_BACKOFF_SECONDS,
        exp_base=BACKOFF_FACTOR,
    ),
    retry=retry_if_exception_type(ResendError),
)
def add_contact_to_resend(
    audience_id: str,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a contact to the Resend audience with retry logic.

    Args:
        audience_id: The Resend audience ID.
        email: The email address of the contact.
        first_name: The first name of the contact.
        last_name: The last name of the contact.

    Returns:
        The API response.

    Raises:
        ResendAPIError: If the API call fails after retries.
    """
    try:
        params = {'audience_id': audience_id, 'email': email}

        if first_name:
            params['first_name'] = first_name

        if last_name:
            params['last_name'] = last_name

        return resend.Contacts.create(params)
    except Exception:
        logger.exception(f'Failed to add contact {email} to Resend')
        raise


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(
        multiplier=INITIAL_BACKOFF_SECONDS,
        max=MAX_BACKOFF_SECONDS,
        exp_base=BACKOFF_FACTOR,
    ),
    retry=retry_if_exception_type(ResendError),
)
def send_welcome_email(
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a welcome email to a new contact.

    Args:
        email: The email address of the contact.
        first_name: The first name of the contact.
        last_name: The last name of the contact.

    Returns:
        The API response.

    Raises:
        ResendError: If the API call fails after retries.
    """
    try:
        # Prepare the recipient name
        recipient_name = ''
        if first_name:
            recipient_name = first_name
            if last_name:
                recipient_name += f' {last_name}'

        # Personalize greeting based on available information
        greeting = f'Hi {recipient_name},' if recipient_name else 'Hi there,'

        # Prepare email parameters
        params = {
            'from': os.environ.get(
                'RESEND_FROM_EMAIL', 'OpenHands Team <no-reply@welcome.openhands.dev>'
            ),
            'reply_to': os.environ.get(
                'RESEND_REPLY_TO_EMAIL', 'contact@openhands.dev'
            ),
            'to': [email],
            'subject': 'Welcome to OpenHands Cloud',
            'html': f"""
            <div>
                <p>{greeting}</p>
                <p>Thanks for joining OpenHands Cloud — we're excited to help you start building with the world's leading open source AI coding agent!</p>
                <p><strong>Here are three quick ways to get started:</strong></p>
                <ol>
                    <li><a href="https://docs.openhands.dev/openhands/usage/cloud/openhands-cloud#next-steps"><strong>Connect your Git repo</strong></a> – Link your <a href="https://docs.openhands.dev/openhands/usage/cloud/github-installation">GitHub</a> or <a href="https://docs.openhands.dev/openhands/usage/cloud/gitlab-installation">GitLab</a> repository in seconds so OpenHands can begin understanding your codebase and suggest tasks.</li>
                    <li><a href="https://docs.openhands.dev/openhands/usage/cloud/github-installation#working-on-github-issues-and-pull-requests-using-openhands"><strong>Use OpenHands on an issue or pull request</strong></a> – Label an issue with 'openhands' or mention @openhands on any PR comment to generate explanations, tests, refactors, or doc fixes tailored to the exact lines you're reviewing.</li>
                    <li><a href="https://dub.sh/openhands"><strong>Join the community</strong></a> – Join our Slack Community to share tips, feedback, and help shape the next features on our roadmap.</li>
                </ol>
                <p>Have questions? Want to share feedback? Just reply to this email—we're here to help.</p>
                <p>Happy coding!</p>
                <p>The OpenHands team</p>
                <p>--</p>
                <p>OpenHands</p>
                <p>24 Oak Street</p>
                <p>Cambridge MA 02139</p>
                <p>https://openhands.dev</p>
            </div>
            """,
        }

        # Send the email
        response = resend.Emails.send(params)
        logger.info(f'Welcome email sent to {email}')
        return response
    except Exception:
        logger.exception(f'Failed to send welcome email to {email}')
        raise


def _get_resend_synced_user_store() -> ResendSyncedUserStore:
    """Get the ResendSyncedUserStore instance.

    This is separated into a function to allow for easier testing/mocking.
    """
    return ResendSyncedUserStore(session_maker=_get_session_maker())


def _backfill_existing_resend_contacts(
    synced_user_store: ResendSyncedUserStore,
    audience_id: str,
) -> int:
    """Backfill the synced_users table with contacts already in Resend.

    This ensures that users who were added to Resend before the tracking
    table existed are properly recorded, preventing duplicate welcome emails.

    Args:
        synced_user_store: The store for tracking synced users.
        audience_id: The Resend audience ID.

    Returns:
        The number of contacts backfilled.
    """
    logger.info('Starting backfill of existing Resend contacts...')

    try:
        resend_contacts = get_resend_contacts(audience_id)
        logger.info(f'Found {len(resend_contacts)} contacts in Resend audience')

        already_synced_emails = synced_user_store.get_synced_emails_for_audience(
            audience_id
        )
        logger.info(
            f'Found {len(already_synced_emails)} already synced emails in database'
        )

        backfilled_count = 0
        for email in resend_contacts:
            if email.lower() not in already_synced_emails:
                synced_user_store.mark_user_synced(
                    email=email,
                    audience_id=audience_id,
                    user_id=None,  # Backfilled Resend contacts have no local user ID.
                )
                backfilled_count += 1
                logger.debug(f'Backfilled existing Resend contact: {email}')

        logger.info(
            f'Backfill completed: {backfilled_count} contacts added to tracking'
        )
        return backfilled_count

    except Exception:
        logger.exception('Error during backfill of existing Resend contacts')
        # Don't fail the entire sync if backfill fails - just log and continue
        return 0


def sync_users_to_resend():
    """Sync OpenHands users to Resend.

    This function syncs users from the OpenHands database to a Resend audience.
    It tracks which users have been synced in the database to ensure that:
    1. Users are only added once (even across multiple sync runs)
    2. Users who are manually deleted from Resend are not re-added

    The tracking is done via the resend_synced_users table, which records
    each email/audience_id combination that has been synced.

    On first run (or when new contacts exist in Resend), it will backfill
    the tracking table with existing Resend contacts to avoid sending
    duplicate welcome emails.
    """
    required_vars = {
        'RESEND_API_KEY': RESEND_API_KEY,
        'RESEND_AUDIENCE_ID': RESEND_AUDIENCE_ID,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        for var in missing_vars:
            logger.error(f'{var} environment variable is not set')
        sys.exit(1)

    logger.info(
        f'Starting sync of OpenHands users to Resend audience {RESEND_AUDIENCE_ID}'
    )

    try:
        synced_user_store = _get_resend_synced_user_store()

        # Backfill existing Resend contacts into our tracking table.
        # This ensures users already in Resend don't get duplicate welcome emails.
        backfilled_count = _backfill_existing_resend_contacts(
            synced_user_store, RESEND_AUDIENCE_ID
        )

        total_users = get_total_local_users()
        logger.info(f'Found {total_users} users with emails in OpenHands database')

        stats = {
            'total_users': total_users,
            'backfilled_contacts': backfilled_count,
            'already_synced': 0,
            'added_contacts': 0,
            'skipped_invalid_emails': 0,
            'errors': 0,
        }

        synced_emails = synced_user_store.get_synced_emails_for_audience(
            RESEND_AUDIENCE_ID
        )
        logger.info(f'Found {len(synced_emails)} already synced emails in database')

        offset = 0
        while offset < total_users:
            users = get_local_users(offset, BATCH_SIZE)
            logger.info(f'Processing batch of {len(users)} users (offset {offset})')

            for user in users:
                email = user.email.lower()

                if email in synced_emails:
                    logger.debug(
                        f'User {email} was already synced to this audience, skipping'
                    )
                    stats['already_synced'] += 1
                    continue

                if not is_valid_email(email):
                    logger.warning(f'Skipping user with invalid email format: {email}')
                    stats['skipped_invalid_emails'] += 1
                    continue

                try:
                    synced_user_store.mark_user_synced(
                        email=email,
                        audience_id=RESEND_AUDIENCE_ID,
                        user_id=user.id,
                    )
                except Exception:
                    logger.exception(f'Failed to mark user {email} as synced')
                    stats['errors'] += 1
                    continue

                try:
                    add_contact_to_resend(
                        RESEND_AUDIENCE_ID, email, user.first_name, user.last_name
                    )
                    logger.info(f'Added user {email} to Resend')
                except Exception:
                    logger.exception(f'Error adding user {email} to Resend')
                    synced_user_store.remove_synced_user(email, RESEND_AUDIENCE_ID)
                    stats['errors'] += 1
                    continue

                synced_emails.add(email)
                stats['added_contacts'] += 1

                time.sleep(1 / RATE_LIMIT)

                try:
                    send_welcome_email(email, user.first_name, user.last_name)
                    logger.info(f'Sent welcome email to {email}')
                except Exception:
                    logger.exception(
                        f'Failed to send welcome email to {email}, '
                        'but contact was added to audience'
                    )

                time.sleep(1 / RATE_LIMIT)

            offset += BATCH_SIZE

        logger.info(f'Sync completed: {stats}')
    except ResendAPIError:
        logger.exception('Resend API error')
        sys.exit(1)
    except Exception:
        logger.exception('Sync failed with unexpected error')
        sys.exit(1)


if __name__ == '__main__':
    sync_users_to_resend()
