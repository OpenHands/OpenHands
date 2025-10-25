"""Test CLI confirm display with dynamic height for model selection."""

from unittest.mock import MagicMock, patch

from openhands.cli.tui import cli_confirm
from openhands.core.config import OpenHandsConfig
from openhands.events.action import ActionSecurityRisk


class TestCliConfirmDynamicHeight:
    """Test that cli_confirm properly handles dynamic height based on number of choices."""

    @patch('openhands.cli.tui.Application')
    def test_cli_confirm_dynamic_height_few_choices(self, mock_app_class):
        """Test height calculation with few choices."""
        config = OpenHandsConfig()
        choices = ['Choice 1', 'Choice 2']

        mock_app = MagicMock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app

        cli_confirm(config, 'Test question?', choices)

        # Verify Application was called with correct layout
        mock_app_class.assert_called_once()
        call_kwargs = mock_app_class.call_args[1]
        layout = call_kwargs['layout']

        # Extract the content window from the layout
        # Layout -> HSplit -> Window (or Frame -> Window for HIGH risk)
        hsplit = layout.container
        window = hsplit.get_children()[0]

        # For non-HIGH risk, window is directly in HSplit
        # For few choices (2), expected height = 3 + 2 + 2 = 7
        expected_height = 7
        assert window.height.preferred == expected_height
        assert window.height.max == expected_height

    @patch('openhands.cli.tui.Application')
    def test_cli_confirm_dynamic_height_many_choices(self, mock_app_class):
        """Test height calculation with many choices (OpenHands models case)."""
        config = OpenHandsConfig()
        # Simulate 12 OpenHands models
        choices = [
            'claude-sonnet-4-20250514',
            'claude-sonnet-4-5-20250929',
            'gpt-5-2025-08-07',
            'gpt-5-mini-2025-08-07',
            'claude-opus-4-20250514',
            'claude-opus-4-1-20250805',
            'devstral-small-2507',
            'devstral-medium-2507',
            'o3',
            'o4-mini',
            'gemini-2.5-pro',
            'kimi-k2-0711-preview',
        ]

        mock_app = MagicMock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app

        cli_confirm(config, 'Select Available OpenHands Model:', choices)

        # Verify Application was called with correct layout
        mock_app_class.assert_called_once()
        call_kwargs = mock_app_class.call_args[1]
        layout = call_kwargs['layout']

        # Extract the content window from the layout
        hsplit = layout.container
        window = hsplit.get_children()[0]

        # For 12 choices, expected height = 3 + 12 + 2 = 17
        expected_height = 17
        assert window.height.preferred == expected_height
        assert window.height.max == expected_height

    @patch('openhands.cli.tui.Application')
    def test_cli_confirm_height_capped_at_20(self, mock_app_class):
        """Test that height is capped at 20 lines even with many choices."""
        config = OpenHandsConfig()
        # Create 25 choices which would exceed the cap
        choices = [f'Choice {i}' for i in range(25)]

        mock_app = MagicMock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app

        cli_confirm(config, 'Test question?', choices)

        # Verify Application was called with correct layout
        mock_app_class.assert_called_once()
        call_kwargs = mock_app_class.call_args[1]
        layout = call_kwargs['layout']

        # Extract the content window from the layout
        hsplit = layout.container
        window = hsplit.get_children()[0]

        # Height should be capped at 20, not 3 + 25 + 2 = 30
        expected_height = 20
        assert window.height.preferred == expected_height
        assert window.height.max == expected_height

    @patch('openhands.cli.tui.Application')
    def test_cli_confirm_high_risk_has_frame(self, mock_app_class):
        """Test that HIGH risk commands work with Frame wrapper."""
        config = OpenHandsConfig()
        choices = ['Yes, proceed', 'No, cancel']

        mock_app = MagicMock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app

        # Just verify it doesn't crash with HIGH risk
        result = cli_confirm(
            config, 'HIGH RISK ACTION?', choices, security_risk=ActionSecurityRisk.HIGH
        )

        # Verify it completed successfully
        assert result == 0
        mock_app_class.assert_called_once()

        # Verify layout has the Frame structure (HSplit with nested children)
        call_kwargs = mock_app_class.call_args[1]
        layout = call_kwargs['layout']
        hsplit = layout.container
        # For HIGH risk with Frame, the first child is a complex HSplit structure
        frame_structure = hsplit.get_children()[0]
        # Frame widget creates an HSplit with multiple children (borders, etc.)
        assert len(frame_structure.get_children()) > 1
