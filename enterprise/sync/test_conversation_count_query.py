#!/usr/bin/env python3
"""Test script to verify the conversation count query.

This script tests the database query to count conversations by user,
without making any API calls to Common Room.
"""

import os
import sys

from sqlalchemy import text

# Add the parent directory to the path so we can import from storage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import get_engine


def test_conversation_count_query():
    """Test the query to count conversations by user."""
    try:
        # Query to count conversations by user
        count_query = text("""
            SELECT
                user_id, COUNT(*) as conversation_count
            FROM
                conversation_metadata
            GROUP BY
                user_id
        """)

        engine = get_engine()

        with engine.connect() as connection:
            count_result = connection.execute(count_query)
            user_counts = [
                {'user_id': row[0], 'conversation_count': row[1]}
                for row in count_result
            ]

        print(f'Found {len(user_counts)} users with conversations')

        # Print the first 5 results
        for i, user_data in enumerate(user_counts[:5]):
            print(
                f"User {i+1}: {user_data['user_id']} - {user_data['conversation_count']} conversations"
            )

        # Test the user_entity query for the first user (if any)
        if user_counts:
            first_user_id = user_counts[0]['user_id']

            user_query = text("""
                SELECT username, email, id
                FROM user_entity
                WHERE id = :user_id
            """)

            with engine.connect() as connection:
                user_result = connection.execute(user_query, {'user_id': first_user_id})
                user_row = user_result.fetchone()

                if user_row:
                    print(f'\nUser details for {first_user_id}:')
                    print(f'  GitHub Username: {user_row[0]}')
                    print(f'  Email: {user_row[1]}')
                    print(f'  ID: {user_row[2]}')
                else:
                    print(
                        f'\nNo user details found for {first_user_id} in user_entity table'
                    )

        print('\nTest completed successfully')
    except Exception as e:
        print(f'Error: {str(e)}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_conversation_count_query()
