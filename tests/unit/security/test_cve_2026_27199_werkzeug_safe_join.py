"""Test for CVE-2026-27199: Werkzeug safe_join Windows device names vulnerability.

This test verifies that werkzeug 3.1.6 properly validates against Windows device
names in multi-segment paths. The vulnerability allowed Windows special device names
(NUL, CON, PRN, etc.) to bypass validation when preceded by other path segments
(e.g., "example/NUL").

On Windows, accessing these special device names would cause the process to hang
indefinitely, resulting in a denial of service.

CVE: https://nvd.nist.gov/vuln/detail/CVE-2026-27199
Advisory: https://github.com/pallets/werkzeug/security/advisories/GHSA-29vq-49wr-vm6x
"""

from unittest.mock import patch

import pytest
from werkzeug.security import safe_join

# Windows special device names that can cause DoS if not properly validated
WINDOWS_DEVICE_NAMES = [
    'NUL',
    'CON',
    'PRN',
    'AUX',
    'CONIN$',
    'CONOUT$',
    'COM1',
    'COM2',
    'COM3',
    'COM4',
    'COM5',
    'COM6',
    'COM7',
    'COM8',
    'COM9',
    'LPT1',
    'LPT2',
    'LPT3',
    'LPT4',
    'LPT5',
    'LPT6',
    'LPT7',
    'LPT8',
    'LPT9',
]


class TestCVE2026_27199:
    """Tests for CVE-2026-27199 fix in werkzeug 3.1.6.

    The vulnerability allowed Windows device names in multi-segment paths to
    bypass validation. For example, "example/NUL" would pass validation in
    werkzeug < 3.1.6 but would cause a denial of service on Windows when
    trying to open the file.
    """

    @pytest.mark.parametrize('device_name', WINDOWS_DEVICE_NAMES)
    def test_rejects_device_names_in_multi_segment_paths_on_windows(
        self, device_name: str
    ) -> None:
        """Verify safe_join rejects Windows device names in multi-segment paths.

        This is the core vulnerability (CVE-2026-27199): device names preceded by
        other path segments (e.g., "subdir/NUL") were not properly validated in
        werkzeug < 3.1.6.
        """
        # Simulate Windows environment where os.name == 'nt'
        with patch('werkzeug.security.os.name', 'nt'):
            # Test multi-segment path with device name at the end
            result = safe_join('/base', f'example/{device_name}')
            assert result is None, (
                f"safe_join should reject 'example/{device_name}' on Windows "
                f'but returned {result}'
            )

            # Test nested multi-segment path
            result = safe_join('/base', f'dir/subdir/{device_name}')
            assert result is None, (
                f"safe_join should reject 'dir/subdir/{device_name}' on Windows "
                f'but returned {result}'
            )

    @pytest.mark.parametrize('device_name', WINDOWS_DEVICE_NAMES)
    def test_rejects_device_names_at_root_level_on_windows(
        self, device_name: str
    ) -> None:
        """Verify safe_join rejects device names at the root level on Windows.

        This was fixed in werkzeug 3.1.4 but we verify it still works.
        """
        with patch('werkzeug.security.os.name', 'nt'):
            result = safe_join('/base', device_name)
            assert result is None, (
                f"safe_join should reject '{device_name}' on Windows "
                f'but returned {result}'
            )

    @pytest.mark.parametrize('device_name', WINDOWS_DEVICE_NAMES)
    def test_rejects_device_names_with_extensions_on_windows(
        self, device_name: str
    ) -> None:
        """Verify safe_join rejects device names even with file extensions.

        Windows device names remain special even with extensions (e.g., NUL.txt).
        This was fixed in werkzeug 3.1.5.
        """
        with patch('werkzeug.security.os.name', 'nt'):
            # Test with extension at root level
            result = safe_join('/base', f'{device_name}.txt')
            assert result is None, (
                f"safe_join should reject '{device_name}.txt' on Windows "
                f'but returned {result}'
            )

            # Test with extension in multi-segment path
            result = safe_join('/base', f'example/{device_name}.txt')
            assert result is None, (
                f"safe_join should reject 'example/{device_name}.txt' on Windows "
                f'but returned {result}'
            )

    @pytest.mark.parametrize('device_name', WINDOWS_DEVICE_NAMES)
    def test_case_insensitive_device_name_detection_on_windows(
        self, device_name: str
    ) -> None:
        """Verify device name detection is case-insensitive on Windows.

        Windows device names are case-insensitive, so NUL, nul, Nul, etc.
        should all be rejected.
        """
        with patch('werkzeug.security.os.name', 'nt'):
            # Test lowercase
            result = safe_join('/base', f'example/{device_name.lower()}')
            assert result is None, (
                f"safe_join should reject 'example/{device_name.lower()}' on Windows "
                f'but returned {result}'
            )

            # Test mixed case
            mixed_case = device_name[0] + device_name[1:].lower()
            result = safe_join('/base', f'example/{mixed_case}')
            assert result is None, (
                f"safe_join should reject 'example/{mixed_case}' on Windows "
                f'but returned {result}'
            )

    def test_allows_normal_paths_on_windows(self) -> None:
        """Verify safe_join allows normal file paths on Windows."""
        with patch('werkzeug.security.os.name', 'nt'):
            # Normal single segment
            result = safe_join('/base', 'file.txt')
            assert result == '/base/file.txt'

            # Normal multi-segment
            result = safe_join('/base', 'dir/subdir/file.txt')
            assert result == '/base/dir/subdir/file.txt'

            # Names that start with device names but aren't device names
            result = safe_join('/base', 'NULLIFY.txt')
            assert result == '/base/NULLIFY.txt'

            result = safe_join('/base', 'CONSOLE.txt')
            assert result == '/base/CONSOLE.txt'

    def test_allows_device_names_on_non_windows(self) -> None:
        """Verify safe_join allows device name paths on non-Windows systems.

        Linux/macOS don't have the Windows device name issue, so these paths
        should be allowed.
        """
        with patch('werkzeug.security.os.name', 'posix'):
            # Device names should be allowed on non-Windows
            result = safe_join('/base', 'NUL')
            assert result == '/base/NUL'

            result = safe_join('/base', 'example/NUL')
            assert result == '/base/example/NUL'

            result = safe_join('/base', 'example/CON.txt')
            assert result == '/base/example/CON.txt'
