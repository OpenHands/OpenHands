"""Tests for the enterprise storage.database module.

These tests verify that the session_maker function properly forwards
keyword arguments to the underlying session maker for backward compatibility.
"""

from unittest.mock import MagicMock, patch


class TestSessionMaker:
    """Test cases for the session_maker function."""

    @patch('enterprise.storage.database._get_db_session_injector')
    def test_session_maker_without_args(self, mock_get_injector):
        """Test that session_maker works without any arguments."""
        from enterprise.storage.database import session_maker

        # Set up mock
        mock_injector = MagicMock()
        mock_inner_session_maker = MagicMock()
        mock_session = MagicMock()
        mock_inner_session_maker.return_value = mock_session
        mock_injector.get_session_maker.return_value = mock_inner_session_maker
        mock_get_injector.return_value = mock_injector

        # Call session_maker without arguments
        result = session_maker()

        # Verify the inner session maker was called without arguments
        mock_inner_session_maker.assert_called_once_with()
        assert result == mock_session

    @patch('enterprise.storage.database._get_db_session_injector')
    def test_session_maker_with_expire_on_commit_false(self, mock_get_injector):
        """Test that session_maker accepts expire_on_commit keyword argument.

        This is a critical backward compatibility test - the session_maker
        must accept keyword arguments like expire_on_commit=False which is
        used in slack.py and potentially other integration modules.
        """
        from enterprise.storage.database import session_maker

        # Set up mock
        mock_injector = MagicMock()
        mock_inner_session_maker = MagicMock()
        mock_session = MagicMock()
        mock_inner_session_maker.return_value = mock_session
        mock_injector.get_session_maker.return_value = mock_inner_session_maker
        mock_get_injector.return_value = mock_injector

        # Call session_maker with expire_on_commit=False
        # This is the exact call pattern used in slack.py line 242
        result = session_maker(expire_on_commit=False)

        # Verify the inner session maker was called with the keyword argument
        mock_inner_session_maker.assert_called_once_with(expire_on_commit=False)
        assert result == mock_session

    @patch('enterprise.storage.database._get_db_session_injector')
    def test_session_maker_with_multiple_kwargs(self, mock_get_injector):
        """Test that session_maker passes through multiple keyword arguments."""
        from enterprise.storage.database import session_maker

        # Set up mock
        mock_injector = MagicMock()
        mock_inner_session_maker = MagicMock()
        mock_session = MagicMock()
        mock_inner_session_maker.return_value = mock_session
        mock_injector.get_session_maker.return_value = mock_inner_session_maker
        mock_get_injector.return_value = mock_injector

        # Call with multiple kwargs
        result = session_maker(
            expire_on_commit=False, autoflush=False, autocommit=False
        )

        # Verify all kwargs were passed through
        mock_inner_session_maker.assert_called_once_with(
            expire_on_commit=False, autoflush=False, autocommit=False
        )
        assert result == mock_session

    @patch('enterprise.storage.database._get_db_session_injector')
    def test_session_maker_returns_correct_session(self, mock_get_injector):
        """Test that session_maker returns the session from the inner session maker."""
        from enterprise.storage.database import session_maker

        # Set up mock
        mock_injector = MagicMock()
        mock_inner_session_maker = MagicMock()
        mock_session = MagicMock()
        mock_inner_session_maker.return_value = mock_session
        mock_injector.get_session_maker.return_value = mock_inner_session_maker
        mock_get_injector.return_value = mock_injector

        result = session_maker()

        # Verify the returned session is from the inner session maker
        assert result is mock_session
