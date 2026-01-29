#!/usr/bin/env python
"""
Downgrade script for migrated users.

This script identifies users who have been migrated (already_migrated=True)
and reverts them back to the pre-migration state.

Usage:
    # Dry run - just list the users that would be downgraded
    python downgrade_migrated_users.py --dry-run

    # Downgrade a specific user by their keycloak_user_id
    python downgrade_migrated_users.py --user-id <user_id>

    # Downgrade all migrated users (with confirmation)
    python downgrade_migrated_users.py --all

    # Downgrade all migrated users without confirmation (dangerous!)
    python downgrade_migrated_users.py --all --no-confirm
"""

import argparse
import asyncio
import sys

# Add the enterprise directory to the path
sys.path.insert(0, '/workspace/project/OpenHands/enterprise')

from server.logger import logger
from sqlalchemy import select, text
from storage.database import session_maker
from storage.user_settings import UserSettings
from storage.user_store import UserStore


def get_migrated_users() -> list[str]:
    """Get list of keycloak_user_ids for users who have been migrated.

    This includes:
    1. Users with already_migrated=True in user_settings (migrated users)
    2. Users in the 'user' table who don't have a user_settings entry (new sign-ups)
    """
    with session_maker() as session:
        # Get users from user_settings with already_migrated=True
        migrated_result = session.execute(
            select(UserSettings.keycloak_user_id).where(
                UserSettings.already_migrated.is_(True)
            )
        )
        migrated_users = {row[0] for row in migrated_result.fetchall() if row[0]}

        # Get users from the 'user' table (new sign-ups won't have user_settings)
        # These are users who signed up after the migration was deployed
        new_signup_result = session.execute(
            text("""
                SELECT CAST(u.id AS VARCHAR)
                FROM "user" u
                WHERE NOT EXISTS (
                    SELECT 1 FROM user_settings us
                    WHERE us.keycloak_user_id = CAST(u.id AS VARCHAR)
                )
            """)
        )
        new_signups = {row[0] for row in new_signup_result.fetchall() if row[0]}

        # Combine both sets
        all_users = migrated_users | new_signups
        return list(all_users)


async def downgrade_user(user_id: str) -> bool:
    """Downgrade a single user.

    Args:
        user_id: The keycloak_user_id to downgrade

    Returns:
        True if successful, False otherwise
    """
    try:
        result = await UserStore.downgrade_user(user_id)
        if result:
            print(f'✓ Successfully downgraded user: {user_id}')
            return True
        else:
            print(f'✗ Failed to downgrade user: {user_id}')
            return False
    except Exception as e:
        print(f'✗ Error downgrading user {user_id}: {e}')
        logger.exception(
            'downgrade_script:error',
            extra={'user_id': user_id, 'error': str(e)},
        )
        return False


async def main():
    parser = argparse.ArgumentParser(
        description='Downgrade migrated users back to pre-migration state'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Just list users that would be downgraded, without making changes',
    )
    parser.add_argument(
        '--user-id',
        type=str,
        help='Downgrade a specific user by keycloak_user_id',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Downgrade all migrated users',
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt (use with caution!)',
    )

    args = parser.parse_args()

    # Get list of migrated users
    migrated_users = get_migrated_users()
    print(f'\nFound {len(migrated_users)} migrated user(s).')

    if args.dry_run:
        print('\n--- DRY RUN MODE ---')
        print('The following users would be downgraded:')
        for user_id in migrated_users:
            print(f'  - {user_id}')
        print('\nNo changes were made.')
        return

    if args.user_id:
        # Downgrade a specific user
        if args.user_id not in migrated_users:
            print(f'\nUser {args.user_id} is not in the migrated users list.')
            print('Either the user was not migrated, or the user_id is incorrect.')
            return

        print(f'\nDowngrading user: {args.user_id}')
        if not args.no_confirm:
            confirm = input('Are you sure? (yes/no): ')
            if confirm.lower() != 'yes':
                print('Cancelled.')
                return

        success = await downgrade_user(args.user_id)
        if success:
            print('\nDowngrade completed successfully.')
        else:
            print('\nDowngrade failed. Check logs for details.')
            sys.exit(1)

    elif args.all:
        # Downgrade all migrated users
        if not migrated_users:
            print('\nNo migrated users to downgrade.')
            return

        print(f'\n⚠️  About to downgrade {len(migrated_users)} user(s).')
        if not args.no_confirm:
            print('\nThis will:')
            print('  - Revert LiteLLM team/user budget settings')
            print('  - Delete organization entries')
            print('  - Delete user entries in the new schema')
            print('  - Reset the already_migrated flag')
            print('\nUsers to downgrade:')
            for user_id in migrated_users[:10]:  # Show first 10
                print(f'  - {user_id}')
            if len(migrated_users) > 10:
                print(f'  ... and {len(migrated_users) - 10} more')

            confirm = input('\nType "yes" to proceed: ')
            if confirm.lower() != 'yes':
                print('Cancelled.')
                return

        print('\nStarting downgrade...\n')
        success_count = 0
        fail_count = 0

        for user_id in migrated_users:
            success = await downgrade_user(user_id)
            if success:
                success_count += 1
            else:
                fail_count += 1

        print('\n--- Summary ---')
        print(f'Successful: {success_count}')
        print(f'Failed: {fail_count}')

        if fail_count > 0:
            sys.exit(1)

    else:
        parser.print_help()
        print('\nPlease specify --dry-run, --user-id, or --all')


if __name__ == '__main__':
    asyncio.run(main())
