#!/usr/bin/env python3
"""Test script for Common Room conversation count sync.

This script tests the functionality of the Common Room sync script
without making any API calls to Common Room or database connections.
"""

import os

# Import the module to test
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sync.common_room_sync import (
    CommonRoomAPIError,
    retry_with_backoff,
    update_common_room_signal,
)


class TestCommonRoomSync(unittest.TestCase):
    """Test cases for Common Room sync functionality."""

    def test_retry_with_backoff(self):
        """Test the retry_with_backoff function."""
        # Mock function that succeeds on the second attempt
        mock_func = MagicMock(
            side_effect=[Exception('First attempt failed'), 'success']
        )

        # Set environment variables for testing
        with patch.dict(
            os.environ,
            {
                'MAX_RETRIES': '3',
                'INITIAL_BACKOFF_SECONDS': '0.01',
                'BACKOFF_FACTOR': '2',
                'MAX_BACKOFF_SECONDS': '1',
            },
        ):
            result = retry_with_backoff(mock_func, 'arg1', 'arg2', kwarg1='kwarg1')

            # Check that the function was called twice
            self.assertEqual(mock_func.call_count, 2)
            # Check that the function was called with the correct arguments
            mock_func.assert_called_with('arg1', 'arg2', kwarg1='kwarg1')
            # Check that the function returned the expected result
            self.assertEqual(result, 'success')

    @patch('sync.common_room_sync.requests.post')
    @patch('sync.common_room_sync.COMMON_ROOM_API_KEY', 'test_api_key')
    @patch(
        'sync.common_room_sync.COMMON_ROOM_DESTINATION_SOURCE_ID',
        'test_source_id',
    )
    def test_update_common_room_signal(self, mock_post):
        """Test the update_common_room_signal function."""
        # Mock successful API responses
        mock_user_response = MagicMock()
        mock_user_response.status_code = 200
        mock_user_response.json.return_value = {'id': 'user123'}

        mock_activity_response = MagicMock()
        mock_activity_response.status_code = 200
        mock_activity_response.json.return_value = {'id': 'activity123'}

        mock_post.side_effect = [mock_user_response, mock_activity_response]

        # Call the function
        result = update_common_room_signal(
            user_id='user123',
            email='user@example.com',
            github_username='user123',
            conversation_count=5,
        )

        # Check that the function made the expected API calls
        self.assertEqual(mock_post.call_count, 2)

        # Check the first call (user creation)
        args1, kwargs1 = mock_post.call_args_list[0]
        self.assertIn('/source/test_source_id/user', args1[0])
        self.assertEqual(kwargs1['headers']['Authorization'], 'Bearer test_api_key')
        self.assertEqual(kwargs1['json']['id'], 'user123')
        self.assertEqual(kwargs1['json']['email'], 'user@example.com')

        # Check the second call (activity creation)
        args2, kwargs2 = mock_post.call_args_list[1]
        self.assertIn('/source/test_source_id/activity', args2[0])
        self.assertEqual(kwargs2['headers']['Authorization'], 'Bearer test_api_key')
        self.assertEqual(kwargs2['json']['user']['id'], 'user123')
        self.assertEqual(
            kwargs2['json']['content']['value'], 'User has created 5 conversations'
        )

        # Check the return value
        self.assertEqual(result, {'id': 'activity123'})

    @patch('sync.common_room_sync.requests.post')
    @patch('sync.common_room_sync.COMMON_ROOM_API_KEY', 'test_api_key')
    @patch(
        'sync.common_room_sync.COMMON_ROOM_DESTINATION_SOURCE_ID',
        'test_source_id',
    )
    def test_update_common_room_signal_error(self, mock_post):
        """Test error handling in update_common_room_signal function."""
        # Mock failed API response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response

        # Call the function and check that it raises the expected exception
        with self.assertRaises(CommonRoomAPIError):
            update_common_room_signal(
                user_id='user123',
                email='user@example.com',
                github_username='user123',
                conversation_count=5,
            )


if __name__ == '__main__':
    unittest.main()
