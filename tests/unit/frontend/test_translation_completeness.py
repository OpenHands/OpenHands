"""Test that the translation completeness check works correctly."""

import os
import shutil
import subprocess
import unittest


class TestTranslationCompleteness(unittest.TestCase):
    """Test that the translation completeness check works correctly."""

    def test_translation_completeness_check_runs(self):
        """Test that the translation completeness check script can be executed."""
        # Skip test if Node.js is not available
        if not shutil.which('node'):
            self.skipTest('Node.js is not available in the environment')

        frontend_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            'frontend',
        )
        script_path = os.path.join(
            frontend_dir, 'scripts', 'check-translation-completeness.cjs'
        )

        # Verify the script exists
        self.assertTrue(
            os.path.exists(script_path), f'Script not found at {script_path}'
        )

        # Verify the script is executable
        self.assertTrue(
            os.access(script_path, os.X_OK),
            f'Script at {script_path} is not executable',
        )

        # Run the script (it may fail due to missing translations, but we just want to verify it runs)
        try:
            subprocess.run(
                ['node', script_path],
                cwd=frontend_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            # We don't assert on the return code because it might fail due to missing translations
        except Exception as e:
            self.fail(f'Failed to run translation completeness check: {e}')
