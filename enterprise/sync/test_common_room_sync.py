#!/usr/bin/env python3
"""
Test script for Common Room conversation count sync.

This script tests the functionality of the Common Room sync script
without making any API calls to Common Room or database connections.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sync.common_room_sync import (
    retry_with_backoff,
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


if __name__ == '__main__':
    unittest.main()
