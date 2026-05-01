import re
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import patch

from openhands.app_server.version import get_version

VERSION_PATTERN = re.compile(r'^\d+\.\d+\.\d+$')


def _write_pyproject(tmp_path: Path, version: str) -> Path:
    pyproject = tmp_path / 'pyproject.toml'
    pyproject.write_text(
        textwrap.dedent(f"""\
            [tool.poetry]
            name = "test-package"
            version = "{version}"
        """)
    )
    return pyproject


class TestGetVersionFromPyproject:
    """Tests for the pyproject.toml-based version resolution."""

    def test_returns_valid_semver_from_real_pyproject(self):
        version = get_version()
        assert VERSION_PATTERN.match(version), (
            f"Expected version matching X.Y.Z (positive ints), got '{version}'"
        )

    def test_reads_first_candidate_path(self, tmp_path):
        _write_pyproject(tmp_path, '1.2.3')
        fake_file = tmp_path / 'openhands' / 'app_server' / 'version.py'
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()

        with patch(
            'openhands.app_server.version.os.path.abspath', return_value=str(fake_file)
        ):
            assert get_version() == '1.2.3'

    def test_reads_second_candidate_path(self, tmp_path):
        # Only place pyproject.toml under the openhands/ subdirectory
        openhands_dir = tmp_path / 'openhands'
        openhands_dir.mkdir()
        _write_pyproject(openhands_dir, '4.5.6')

        fake_file = tmp_path / 'openhands' / 'app_server' / 'version.py'
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()

        with patch(
            'openhands.app_server.version.os.path.abspath', return_value=str(fake_file)
        ):
            assert get_version() == '4.5.6'

    def test_strips_quotes(self, tmp_path):
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text("version = '9.8.7'\n")

        fake_file = tmp_path / 'openhands' / 'app_server' / 'version.py'
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()

        with patch(
            'openhands.app_server.version.os.path.abspath', return_value=str(fake_file)
        ):
            assert get_version() == '9.8.7'


class TestGetVersionFallbacks:
    """Tests for importlib.metadata and pkg_resources fallback paths."""

    def _patch_no_pyproject(self):
        """Patch so that no pyproject.toml candidate files exist."""
        return patch(
            'openhands.app_server.version.os.path.abspath',
            return_value='/nonexistent/openhands/app_server/version.py',
        )

    def test_falls_back_to_importlib_metadata(self):
        with (
            self._patch_no_pyproject(),
            patch('importlib.metadata.version', return_value='10.11.12'),
        ):
            assert get_version() == '10.11.12'

    def test_falls_back_to_pkg_resources(self):
        from importlib.metadata import PackageNotFoundError

        # Create a fake pkg_resources module so the import inside get_version succeeds
        fake_pkg = types.ModuleType('pkg_resources')
        fake_pkg.DistributionNotFound = type('DistributionNotFound', (Exception,), {})
        fake_pkg.get_distribution = lambda name: type(
            'D', (), {'version': '13.14.15'}
        )()

        with (
            self._patch_no_pyproject(),
            patch('importlib.metadata.version', side_effect=PackageNotFoundError('x')),
            patch.dict(sys.modules, {'pkg_resources': fake_pkg}),
        ):
            assert get_version() == '13.14.15'

    def test_returns_unknown_when_all_methods_fail(self):
        from importlib.metadata import PackageNotFoundError

        with (
            self._patch_no_pyproject(),
            patch('importlib.metadata.version', side_effect=PackageNotFoundError('x')),
            patch.dict(sys.modules, {'pkg_resources': None}),  # force ImportError
        ):
            assert get_version() == 'unknown'


class TestModuleLevelVersion:
    """Test that __version__ at module level is well-formed."""

    def test_module_version_matches_pattern(self):
        from openhands.app_server.version import __version__

        assert VERSION_PATTERN.match(__version__), (
            f"Expected __version__ matching X.Y.Z (positive ints), got '{__version__}'"
        )
