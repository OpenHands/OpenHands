#!/usr/bin/env python3
"""
Common Room Sync

This script queries the database to count conversations created by each user,
then creates or updates a signal in Common Room for each user with their
conversation count.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Set

import requests
from sqlalchemy import text

# Add the parent directory to the path so we can import from storage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.auth.token_manager import get_keycloak_admin
from storage.database import get_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('common_room_sync')

# Common Room API configuration
COMMON_ROOM_API_KEY = os.environ.get('COMMON_ROOM_API_KEY')
COMMON_ROOM_DESTINATION_SOURCE_ID = os.environ.get('COMMON_ROOM_DESTINATION_SOURCE_ID')
COMMON_ROOM_API_BASE_URL = 'https://api.commonroom.io/community/v1'

# Sync configuration
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
KEYCLOAK_BATCH_SIZE = int(os.environ.get('KEYCLOAK_BATCH_SIZE', '20'))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
INITIAL_BACKOFF_SECONDS = float(os.environ.get('INITIAL_BACKOFF_SECONDS', '1'))
MAX_BACKOFF_SECONDS = float(os.environ.get('MAX_BACKOFF_SECONDS', '60'))
BACKOFF_FACTOR = float(os.environ.get('BACKOFF_FACTOR', '2'))
RATE_LIMIT = float(os.environ.get('RATE_LIMIT', '2'))  # Requests per second


class CommonRoomSyncError(Exception):
    """Base exception for Common Room sync errors."""


class DatabaseError(CommonRoomSyncError):
    """Exception for database errors."""


class CommonRoomAPIError(CommonRoomSyncError):
    """Exception for Common Room API errors."""


class KeycloakClientError(CommonRoomSyncError):
    """Exception for Keycloak client errors."""


def get_recent_conversations(minutes: int = 60) -> List[Dict[str, Any]]:
    """Get conversations created in the past N minutes.

    Args:
        minutes: Number of minutes to look back for new conversations.

    Returns:
        A list of dictionaries, each containing conversation details.

    Raises:
        DatabaseError: If the database query fails.
    """
    try:
        # Use a different syntax for the interval that works with pg8000
        query = text("""
            SELECT
                conversation_id, user_id, title, created_at
            FROM
                conversation_metadata
            WHERE
                created_at >= NOW() - (INTERVAL '1 minute' * :minutes)
            ORDER BY
                created_at DESC
        """)

        with get_engine().connect() as connection:
            result = connection.execute(query, {'minutes': minutes})
            conversations = [
                {
                    'conversation_id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'created_at': row[3].isoformat() if row[3] else None,
                }
                for row in result
            ]

        logger.info(
            f'Retrieved {len(conversations)} conversations created in the past {minutes} minutes'
        )
        return conversations
    except Exception as e:
        logger.exception(f'Error querying recent conversations: {e}')
        raise DatabaseError(f'Failed to query recent conversations: {e}')


async def get_users_from_keycloak(user_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Get user information from Keycloak for a set of user IDs.

    Args:
        user_ids: A set of user IDs to look up.

    Returns:
        A dictionary mapping user IDs to user information dictionaries.

    Raises:
        KeycloakClientError: If the Keycloak API call fails.
    """
    try:
        # Get Keycloak admin client
        keycloak_admin = get_keycloak_admin()

        # Create a dictionary to store user information
        user_info_dict = {}

        # Convert set to list for easier batching
        user_id_list = list(user_ids)

        # Process user IDs in batches
        for i in range(0, len(user_id_list), KEYCLOAK_BATCH_SIZE):
            batch = user_id_list[i : i + KEYCLOAK_BATCH_SIZE]
            batch_tasks = []

            # Create tasks for each user ID in the batch
            for user_id in batch:
                # Use the Keycloak admin client to get user by ID
                batch_tasks.append(get_user_by_id(keycloak_admin, user_id))

            # Run the batch of tasks concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process the results
            for user_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f'Error getting user {user_id}: {result}')
                    continue

                if result and isinstance(result, dict):
                    user_info_dict[user_id] = {
                        'username': result.get('username'),
                        'email': result.get('email'),
                        'id': result.get('id'),
                    }

        logger.info(
            f'Retrieved information for {len(user_info_dict)} users from Keycloak'
        )
        return user_info_dict

    except Exception as e:
        error_msg = f'Error getting users from Keycloak: {e}'
        logger.exception(error_msg)
        raise KeycloakClientError(error_msg)


async def get_user_by_id(keycloak_admin, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user from Keycloak by ID.

    Args:
        keycloak_admin: The Keycloak admin client.
        user_id: The user ID to look up.

    Returns:
        A dictionary with the user's information, or None if not found.
    """
    try:
        # Use the Keycloak admin client to get user by ID
        user = keycloak_admin.get_user(user_id)
        if user:
            logger.debug(
                f"Found user in Keycloak: {user.get('username')}, {user.get('email')}"
            )
            return user
        else:
            logger.warning(f'User {user_id} not found in Keycloak')
            return None
    except Exception as e:
        logger.warning(f'Error getting user {user_id} from Keycloak: {e}')
        return None


def get_user_info(
    user_id: str, user_info_cache: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, str]]:
    """Get the email address and GitHub username for a user from the cache.

    Args:
        user_id: The user ID to look up.
        user_info_cache: A dictionary mapping user IDs to user information.

    Returns:
        A dictionary with the user's email and username, or None if not found.
    """
    # Check if the user is in the cache
    if user_id in user_info_cache:
        user_info = user_info_cache[user_id]
        logger.debug(
            f"Found user info in cache: {user_info.get('username')}, {user_info.get('email')}"
        )
        return user_info
    else:
        logger.warning(f'User {user_id} not found in user info cache')
        return None


def register_user_in_common_room(
    user_id: str, email: str, github_username: str
) -> Dict[str, Any]:
    """Create or update a user in Common Room.

    Args:
        user_id: The user ID.
        email: The user's email address.
        github_username: The user's GitHub username.

    Returns:
        The API response from Common Room.

    Raises:
        CommonRoomAPIError: If the Common Room API request fails.
    """
    if not COMMON_ROOM_API_KEY:
        raise CommonRoomAPIError('COMMON_ROOM_API_KEY environment variable not set')

    if not COMMON_ROOM_DESTINATION_SOURCE_ID:
        raise CommonRoomAPIError(
            'COMMON_ROOM_DESTINATION_SOURCE_ID environment variable not set'
        )

    try:
        headers = {
            'Authorization': f'Bearer {COMMON_ROOM_API_KEY}',
            'Content-Type': 'application/json',
        }

        # Create or update user in Common Room
        user_data = {
            'id': user_id,
            'email': email,
            'username': github_username,
            'github': {'type': 'handle', 'value': github_username},
        }

        user_url = f'{COMMON_ROOM_API_BASE_URL}/source/{COMMON_ROOM_DESTINATION_SOURCE_ID}/user'
        user_response = requests.post(user_url, headers=headers, json=user_data)

        if user_response.status_code not in (200, 202):
            logger.error(
                f'Failed to create/update user in Common Room: {user_response.text}'
            )
            logger.error(f'Response status code: {user_response.status_code}')
            raise CommonRoomAPIError(
                f'Failed to create/update user: {user_response.text}'
            )

        logger.info(
            f'Registered/updated user {user_id} (GitHub: {github_username}) in Common Room'
        )
        return user_response.json()
    except requests.RequestException as e:
        logger.exception(f'Error communicating with Common Room API: {e}')
        raise CommonRoomAPIError(f'Failed to communicate with Common Room API: {e}')


def register_conversation_activity(
    user_id: str,
    conversation_id: str,
    conversation_title: str,
    created_at: datetime,
    email: str,
    github_username: str,
) -> Dict[str, Any]:
    """Create an activity in Common Room for a new conversation.

    Args:
        user_id: The user ID who created the conversation.
        conversation_id: The ID of the conversation.
        conversation_title: The title of the conversation.
        created_at: The datetime object when the conversation was created.
        email: The user's email address.
        github_username: The user's GitHub username.

    Returns:
        The API response from Common Room.

    Raises:
        CommonRoomAPIError: If the Common Room API request fails.
    """
    if not COMMON_ROOM_API_KEY:
        raise CommonRoomAPIError('COMMON_ROOM_API_KEY environment variable not set')

    if not COMMON_ROOM_DESTINATION_SOURCE_ID:
        raise CommonRoomAPIError(
            'COMMON_ROOM_DESTINATION_SOURCE_ID environment variable not set'
        )

    try:
        headers = {
            'Authorization': f'Bearer {COMMON_ROOM_API_KEY}',
            'Content-Type': 'application/json',
        }

        # Format the datetime object to the expected ISO format
        formatted_timestamp = (
            created_at.strftime('%Y-%m-%dT%H:%M:%SZ')
            if created_at
            else time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        )

        # Create activity for the conversation
        activity_data = {
            'id': f'conversation_{conversation_id}',  # Use conversation ID to ensure uniqueness
            'activityType': 'started_session',
            'user': {
                'id': user_id,
                'email': email,
                'github': {'type': 'handle', 'value': github_username},
                'username': github_username,
            },
            'activityTitle': {
                'type': 'text',
                'value': conversation_title or 'New Conversation',
            },
            'content': {
                'type': 'text',
                'value': f'Started a new conversation: {conversation_title or "Untitled"}',
            },
            'timestamp': formatted_timestamp,
            'url': f'https://app.all-hands.dev/conversations/{conversation_id}',
        }

        # Log the activity data for debugging
        logger.info(f'Activity data payload: {activity_data}')

        activity_url = f'{COMMON_ROOM_API_BASE_URL}/source/{COMMON_ROOM_DESTINATION_SOURCE_ID}/activity'
        activity_response = requests.post(
            activity_url, headers=headers, json=activity_data
        )

        if activity_response.status_code not in (200, 202):
            logger.error(
                f'Failed to create activity in Common Room: {activity_response.text}'
            )
            logger.error(f'Response status code: {activity_response.status_code}')
            raise CommonRoomAPIError(
                f'Failed to create activity: {activity_response.text}'
            )

        logger.info(
            f'Registered conversation activity for user {user_id}, conversation {conversation_id}'
        )
        return activity_response.json()
    except requests.RequestException as e:
        logger.exception(f'Error communicating with Common Room API: {e}')
        raise CommonRoomAPIError(f'Failed to communicate with Common Room API: {e}')


def retry_with_backoff(func, *args, **kwargs):
    """Retry a function with exponential backoff.

    Args:
        func: The function to retry.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.

    Raises:
        The last exception raised by the function.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f'Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}')

            if attempt < MAX_RETRIES - 1:
                sleep_time = min(backoff, MAX_BACKOFF_SECONDS)
                logger.info(f'Retrying in {sleep_time:.2f} seconds...')
                time.sleep(sleep_time)
                backoff *= BACKOFF_FACTOR
            else:
                logger.exception(f'All {MAX_RETRIES} attempts failed')
                raise last_exception


async def retry_with_backoff_async(func, *args, **kwargs):
    """Retry an async function with exponential backoff.

    Args:
        func: The async function to retry.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.

    Raises:
        The last exception raised by the function.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f'Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}')

            if attempt < MAX_RETRIES - 1:
                sleep_time = min(backoff, MAX_BACKOFF_SECONDS)
                logger.info(f'Retrying in {sleep_time:.2f} seconds...')
                await asyncio.sleep(sleep_time)
                backoff *= BACKOFF_FACTOR
            else:
                logger.exception(f'All {MAX_RETRIES} attempts failed')
                raise last_exception


async def async_sync_recent_conversations_to_common_room(minutes: int = 60):
    """Async main function to sync recent conversations to Common Room.

    Args:
        minutes: Number of minutes to look back for new conversations.
    """
    logger.info(
        f'Starting Common Room recent conversations sync (past {minutes} minutes)'
    )

    stats = {
        'total_conversations': 0,
        'registered_users': 0,
        'registered_activities': 0,
        'errors': 0,
        'missing_user_info': 0,
    }

    try:
        # Get conversations created in the past N minutes
        recent_conversations = retry_with_backoff(get_recent_conversations, minutes)
        stats['total_conversations'] = len(recent_conversations)

        logger.info(f'Processing {len(recent_conversations)} recent conversations')

        if not recent_conversations:
            logger.info('No recent conversations found, exiting')
            return

        # Extract all unique user IDs
        user_ids = {conv['user_id'] for conv in recent_conversations if conv['user_id']}

        # Get user information for all users in batches
        user_info_cache = await retry_with_backoff_async(
            get_users_from_keycloak, user_ids
        )

        # Track registered users to avoid duplicate registrations
        registered_users = set()

        # Process each conversation
        for conversation in recent_conversations:
            conversation_id = conversation['conversation_id']
            user_id = conversation['user_id']
            title = conversation['title']
            created_at = conversation[
                'created_at'
            ]  # This might be a string or datetime object

            try:
                # Get user info from cache
                user_info = get_user_info(user_id, user_info_cache)
                if not user_info:
                    logger.warning(
                        f'Could not find user info for user {user_id}, skipping conversation {conversation_id}'
                    )
                    stats['missing_user_info'] += 1
                    continue

                email = user_info['email']
                github_username = user_info['username']

                if not email:
                    logger.warning(
                        f'User {user_id} has no email, skipping conversation {conversation_id}'
                    )
                    stats['errors'] += 1
                    continue

                # Register user in Common Room if not already registered in this run
                if user_id not in registered_users:
                    register_user_in_common_room(user_id, email, github_username)
                    registered_users.add(user_id)
                    stats['registered_users'] += 1

                # If created_at is a string, parse it to a datetime object
                # If it's already a datetime object, use it as is
                # If it's None, use current time
                created_at_datetime = (
                    created_at
                    if isinstance(created_at, datetime)
                    else datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if created_at
                    else datetime.now(UTC)
                )

                # Register conversation activity with email and github username
                register_conversation_activity(
                    user_id,
                    conversation_id,
                    title,
                    created_at_datetime,
                    email,
                    github_username,
                )
                stats['registered_activities'] += 1

                # Sleep to respect rate limit
                await asyncio.sleep(1 / RATE_LIMIT)
            except Exception as e:
                logger.exception(
                    f'Error processing conversation {conversation_id} for user {user_id}: {e}'
                )
                stats['errors'] += 1
    except Exception as e:
        logger.exception(f'Sync failed: {e}')
        raise
    finally:
        logger.info(f'Sync completed. Stats: {stats}')


def sync_recent_conversations_to_common_room(minutes: int = 60):
    """Main function to sync recent conversations to Common Room.

    Args:
        minutes: Number of minutes to look back for new conversations.
    """
    # Run the async function in the event loop
    asyncio.run(async_sync_recent_conversations_to_common_room(minutes))


if __name__ == '__main__':
    # Default to looking back 60 minutes for new conversations
    minutes = int(os.environ.get('SYNC_MINUTES', '60'))
    sync_recent_conversations_to_common_room(minutes)
