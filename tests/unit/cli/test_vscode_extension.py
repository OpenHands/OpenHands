import os
import pathlib
import subprocess
from unittest import mock

import pytest

from openhands.cli import vscode_extension


@pytest.fixture
def mock_env_and_dependencies():
    """A fixture to mock all external dependencies and manage the environment."""
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('pathlib.Path.home') as mock_home,
        mock.patch('pathlib.Path.exists') as mock_exists,
        mock.patch('pathlib.Path.touch') as mock_touch,
        mock.patch('pathlib.Path.mkdir') as mock_mkdir,
        mock.patch('subprocess.run') as mock_subprocess,
        mock.patch('importlib.resources.as_file') as mock_as_file,
        mock.patch(
            'openhands.cli.vscode_extension.download_latest_vsix_from_github'
        ) as mock_download,
        mock.patch('builtins.print') as mock_print,
        mock.patch('openhands.cli.vscode_extension.logger.debug') as mock_logger,
    ):
        # Setup a temporary directory for home
        temp_dir = pathlib.Path.cwd() / 'temp_test_home'
        temp_dir.mkdir(exist_ok=True)
        mock_home.return_value = temp_dir

        try:
            yield {
                'home': mock_home,
                'exists': mock_exists,
                'touch': mock_touch,
                'mkdir': mock_mkdir,
                'subprocess': mock_subprocess,
                'as_file': mock_as_file,
                'download': mock_download,
                'print': mock_print,
                'logger': mock_logger,
            }
        finally:
            # Teardown the temporary directory, ignoring errors if files don't exist
            openhands_dir = temp_dir / '.openhands'
            if openhands_dir.exists():
                for f in openhands_dir.glob('*'):
                    if f.is_file():
                        f.unlink()
                try:
                    openhands_dir.rmdir()
                except FileNotFoundError:
                    pass
            try:
                temp_dir.rmdir()
            except (FileNotFoundError, OSError):
                pass


def test_not_in_vscode_environment(mock_env_and_dependencies):
    """Should not attempt any installation if not in a VSCode-like environment."""
    os.environ['TERM_PROGRAM'] = 'not_vscode'
    vscode_extension.attempt_vscode_extension_install()
    mock_env_and_dependencies['download'].assert_not_called()
    mock_env_and_dependencies['subprocess'].assert_not_called()


def test_already_attempted_flag_prevents_execution(mock_env_and_dependencies):
    """Should do nothing if the installation flag file already exists."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = True  # Simulate flag file exists
    vscode_extension.attempt_vscode_extension_install()
    mock_env_and_dependencies['download'].assert_not_called()
    mock_env_and_dependencies['subprocess'].assert_not_called()


def test_extension_already_installed_detected(mock_env_and_dependencies):
    """Should detect already installed extension and create flag."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    # Mock subprocess call for --list-extensions (returns extension as installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0,
        args=[],
        stdout='openhands.openhands-vscode\nother.extension',
        stderr='',
    )

    vscode_extension.attempt_vscode_extension_install()

    # Should only call --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['subprocess'].assert_called_with(
        ['code', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: OpenHands VS Code extension is already installed.'
    )
    mock_env_and_dependencies['touch'].assert_called_once()
    mock_env_and_dependencies['download'].assert_not_called()


def test_extension_detection_in_middle_of_list(mock_env_and_dependencies):
    """Should detect extension even when it's not the first in the list."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    # Extension is in the middle of the list
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0,
        args=[],
        stdout='first.extension\nopenhands.openhands-vscode\nlast.extension',
        stderr='',
    )

    vscode_extension.attempt_vscode_extension_install()

    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: OpenHands VS Code extension is already installed.'
    )
    mock_env_and_dependencies['touch'].assert_called_once()


def test_extension_detection_partial_match_ignored(mock_env_and_dependencies):
    """Should not match partial extension IDs."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    # Partial match should not trigger detection
    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=0,
            args=[],
            stdout='other.openhands-vscode-fork\nsome.extension',
            stderr='',
        ),
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # Bundled install succeeds
    ]

    # Mock bundled VSIX to succeed
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/fake/path/to/bundled.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    vscode_extension.attempt_vscode_extension_install()

    # Should proceed with installation since exact match not found
    assert mock_env_and_dependencies['subprocess'].call_count == 2
    mock_env_and_dependencies['as_file'].assert_called_once()
    # GitHub download should not be attempted since bundled install succeeds
    mock_env_and_dependencies['download'].assert_not_called()


def test_list_extensions_fails_continues_installation(mock_env_and_dependencies):
    """Should continue with installation if --list-extensions fails."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    # --list-extensions fails, but bundled install succeeds
    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=1, args=[], stdout='', stderr='Command failed'
        ),
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # Bundled install succeeds
    ]

    # Mock bundled VSIX to succeed
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/fake/path/to/bundled.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    vscode_extension.attempt_vscode_extension_install()

    # Should proceed with installation
    assert mock_env_and_dependencies['subprocess'].call_count == 2
    mock_env_and_dependencies['as_file'].assert_called_once()
    # GitHub download should not be attempted since bundled install succeeds
    mock_env_and_dependencies['download'].assert_not_called()


def test_list_extensions_exception_continues_installation(mock_env_and_dependencies):
    """Should continue with installation if --list-extensions throws exception."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    # --list-extensions throws exception, but bundled install succeeds
    mock_env_and_dependencies['subprocess'].side_effect = [
        FileNotFoundError('code command not found'),
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # Bundled install succeeds
    ]

    # Mock bundled VSIX to succeed
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/fake/path/to/bundled.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    vscode_extension.attempt_vscode_extension_install()

    # Should proceed with installation
    assert mock_env_and_dependencies['subprocess'].call_count == 2
    mock_env_and_dependencies['as_file'].assert_called_once()
    # GitHub download should not be attempted since bundled install succeeds
    mock_env_and_dependencies['download'].assert_not_called()


def test_mark_installation_successful_os_error(mock_env_and_dependencies):
    """Should log error but continue if flag file creation fails."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    # Mock bundled VSIX to succeed
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/fake/path/to/bundled.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --list-extensions (empty)
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # Bundled install succeeds
    ]
    mock_env_and_dependencies['touch'].side_effect = OSError('Permission denied')

    vscode_extension.attempt_vscode_extension_install()

    # Should still complete installation
    mock_env_and_dependencies['as_file'].assert_called_once()
    # GitHub download should not be attempted since bundled install succeeds
    mock_env_and_dependencies['download'].assert_not_called()
    mock_env_and_dependencies['touch'].assert_called_once()
    # Should log the error
    mock_env_and_dependencies['logger'].assert_any_call(
        'Could not create VS Code extension success flag file: Permission denied'
    )


def test_installation_failure_no_flag_created(mock_env_and_dependencies):
    """Should NOT create flag when all installation methods fail (allow retry)."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0,
        args=[],
        stdout='',
        stderr='',  # --list-extensions (empty)
    )
    mock_env_and_dependencies['download'].return_value = None  # GitHub fails
    mock_env_and_dependencies[
        'as_file'
    ].side_effect = FileNotFoundError  # Bundled fails

    vscode_extension.attempt_vscode_extension_install()

    # Should NOT create flag file - this is the key behavior change
    mock_env_and_dependencies['touch'].assert_not_called()
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Will retry installation next time you run OpenHands in VS Code.'
    )


def test_install_succeeds_from_bundled(mock_env_and_dependencies):
    """Should successfully install from bundled VSIX on the first try."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False

    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/fake/path/to/bundled.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    # Mock subprocess calls: first --list-extensions (returns empty), then install
    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --list-extensions
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --install-extension
    ]

    vscode_extension.attempt_vscode_extension_install()

    mock_env_and_dependencies['as_file'].assert_called_once()
    # Should have two subprocess calls: list-extensions and install-extension
    assert mock_env_and_dependencies['subprocess'].call_count == 2
    mock_env_and_dependencies['subprocess'].assert_any_call(
        ['code', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['subprocess'].assert_any_call(
        ['code', '--install-extension', '/fake/path/to/bundled.vsix', '--force'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Bundled VS Code extension installed successfully.'
    )
    mock_env_and_dependencies['touch'].assert_called_once()
    # GitHub download should not be attempted
    mock_env_and_dependencies['download'].assert_not_called()


def test_bundled_fails_falls_back_to_github(mock_env_and_dependencies):
    """Should fall back to GitHub if bundled VSIX installation fails."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = '/fake/path/to/github.vsix'

    # Mock bundled VSIX to fail
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess calls: first --list-extensions (returns empty), then install
    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --list-extensions
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --install-extension
    ]

    with (
        mock.patch('os.remove') as mock_os_remove,
        mock.patch('os.path.exists', return_value=True),
    ):
        vscode_extension.attempt_vscode_extension_install()

        mock_env_and_dependencies['as_file'].assert_called_once()
        mock_env_and_dependencies['download'].assert_called_once()
        # Should have two subprocess calls: list-extensions and install-extension
        assert mock_env_and_dependencies['subprocess'].call_count == 2
        mock_env_and_dependencies['subprocess'].assert_any_call(
            ['code', '--list-extensions'],
            capture_output=True,
            text=True,
            check=False,
        )
        mock_env_and_dependencies['subprocess'].assert_any_call(
            ['code', '--install-extension', '/fake/path/to/github.vsix', '--force'],
            capture_output=True,
            text=True,
            check=False,
        )
        mock_env_and_dependencies['print'].assert_any_call(
            'INFO: OpenHands VS Code extension installed successfully from GitHub.'
        )
        mock_os_remove.assert_called_once_with('/fake/path/to/github.vsix')
        mock_env_and_dependencies['touch'].assert_called_once()


def test_all_methods_fail(mock_env_and_dependencies):
    """Should show a final failure message if all installation methods fail."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    mock_env_and_dependencies['download'].assert_called_once()
    mock_env_and_dependencies['as_file'].assert_called_once()
    # Only one subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['subprocess'].assert_called_with(
        ['code', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Will retry installation next time you run OpenHands in VS Code.'
    )
    # Should NOT create flag file on failure - that's the point of our new approach
    mock_env_and_dependencies['touch'].assert_not_called()


def test_windsurf_detection_and_install(mock_env_and_dependencies):
    """Should correctly detect Windsurf but not attempt marketplace installation."""
    os.environ['__CFBundleIdentifier'] = 'com.exafunction.windsurf'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # Only one subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['subprocess'].assert_called_with(
        ['surf', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Will retry installation next time you run OpenHands in Windsurf.'
    )
    # Should NOT create flag file on failure
    mock_env_and_dependencies['touch'].assert_not_called()


def test_os_error_on_mkdir(mock_env_and_dependencies):
    """Should log a debug message if creating the flag directory fails."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['mkdir'].side_effect = OSError('Permission denied')

    vscode_extension.attempt_vscode_extension_install()

    mock_env_and_dependencies['logger'].assert_called_once_with(
        'Could not create or check VS Code extension flag directory: Permission denied'
    )
    mock_env_and_dependencies['download'].assert_not_called()


def test_os_error_on_touch(mock_env_and_dependencies):
    """Should log a debug message if creating the flag file fails."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )
    mock_env_and_dependencies['touch'].side_effect = OSError('Permission denied')

    vscode_extension.attempt_vscode_extension_install()

    # Should NOT create flag file on failure - this is the new behavior
    mock_env_and_dependencies['touch'].assert_not_called()
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Will retry installation next time you run OpenHands in VS Code.'
    )


def test_flag_file_exists_windsurf(mock_env_and_dependencies):
    """Should not attempt install if flag file already exists (Windsurf)."""
    os.environ['__CFBundleIdentifier'] = 'com.exafunction.windsurf'
    mock_env_and_dependencies['exists'].return_value = True
    vscode_extension.attempt_vscode_extension_install()
    mock_env_and_dependencies['download'].assert_not_called()
    mock_env_and_dependencies['subprocess'].assert_not_called()


def test_successful_install_attempt_vscode(mock_env_and_dependencies):
    """Test that VS Code is detected but marketplace installation is not attempted."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['subprocess'].assert_called_with(
        ['code', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_successful_install_attempt_windsurf(mock_env_and_dependencies):
    """Test that Windsurf is detected but marketplace installation is not attempted."""
    os.environ['__CFBundleIdentifier'] = 'com.exafunction.windsurf'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['subprocess'].assert_called_with(
        ['surf', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_install_attempt_code_command_fails(mock_env_and_dependencies):
    """Test that VS Code is detected but marketplace installation is not attempted."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_install_attempt_code_not_found(mock_env_and_dependencies):
    """Test that VS Code is detected but marketplace installation is not attempted."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_flag_dir_creation_os_error_windsurf(mock_env_and_dependencies):
    """Test OSError during flag directory creation (Windsurf)."""
    os.environ['__CFBundleIdentifier'] = 'com.exafunction.windsurf'
    mock_env_and_dependencies['mkdir'].side_effect = OSError('Permission denied')
    vscode_extension.attempt_vscode_extension_install()
    mock_env_and_dependencies['logger'].assert_called_once_with(
        'Could not create or check Windsurf extension flag directory: Permission denied'
    )
    mock_env_and_dependencies['download'].assert_not_called()


def test_flag_file_touch_os_error_vscode(mock_env_and_dependencies):
    """Test OSError during flag file touch (VS Code)."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )
    mock_env_and_dependencies['touch'].side_effect = OSError('Permission denied')

    vscode_extension.attempt_vscode_extension_install()

    # Should NOT create flag file on failure - this is the new behavior
    mock_env_and_dependencies['touch'].assert_not_called()
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Will retry installation next time you run OpenHands in VS Code.'
    )


def test_flag_file_touch_os_error_windsurf(mock_env_and_dependencies):
    """Test OSError during flag file touch (Windsurf)."""
    os.environ['__CFBundleIdentifier'] = 'com.exafunction.windsurf'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )
    mock_env_and_dependencies['touch'].side_effect = OSError('Permission denied')

    vscode_extension.attempt_vscode_extension_install()

    # Should NOT create flag file on failure - this is the new behavior
    mock_env_and_dependencies['touch'].assert_not_called()
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Will retry installation next time you run OpenHands in Windsurf.'
    )


def test_bundled_vsix_installation_failure_fallback_to_marketplace(
    mock_env_and_dependencies,
):
    """Test bundled VSIX failure shows appropriate message."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/mock/path/openhands-vscode-0.0.1.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    # Mock subprocess calls: first --list-extensions (empty), then bundled install (fails)
    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --list-extensions
        subprocess.CompletedProcess(
            args=[
                'code',
                '--install-extension',
                '/mock/path/openhands-vscode-0.0.1.vsix',
                '--force',
            ],
            returncode=1,
            stdout='Installation failed',
            stderr='Error installing extension',
        ),
    ]

    vscode_extension.attempt_vscode_extension_install()

    # Two subprocess calls: --list-extensions and bundled VSIX install
    assert mock_env_and_dependencies['subprocess'].call_count == 2
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_bundled_vsix_not_found_fallback_to_marketplace(mock_env_and_dependencies):
    """Test bundled VSIX not found shows appropriate message."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = False
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_importlib_resources_exception_fallback_to_marketplace(
    mock_env_and_dependencies,
):
    """Test importlib.resources exception shows appropriate message."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError(
        'Resource not found'
    )

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_comprehensive_windsurf_detection_path_based(mock_env_and_dependencies):
    """Test Windsurf detection via PATH environment variable but no marketplace installation."""
    os.environ['PATH'] = (
        '/usr/local/bin:/Applications/Windsurf.app/Contents/Resources/app/bin:/usr/bin'
    )
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['subprocess'].assert_called_with(
        ['surf', '--list-extensions'],
        capture_output=True,
        text=True,
        check=False,
    )
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_comprehensive_windsurf_detection_env_value_based(mock_env_and_dependencies):
    """Test Windsurf detection via environment variable values but no marketplace installation."""
    os.environ['SOME_APP_PATH'] = '/Applications/Windsurf.app/Contents/MacOS/Windsurf'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_comprehensive_windsurf_detection_multiple_indicators(
    mock_env_and_dependencies,
):
    """Test Windsurf detection with multiple environment indicators."""
    os.environ['__CFBundleIdentifier'] = 'com.exafunction.windsurf'
    os.environ['PATH'] = (
        '/usr/local/bin:/Applications/Windsurf.app/Contents/Resources/app/bin:/usr/bin'
    )
    os.environ['WINDSURF_CONFIG'] = '/Users/test/.windsurf/config'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    # Mock subprocess call for --list-extensions (returns empty, extension not installed)
    mock_env_and_dependencies['subprocess'].return_value = subprocess.CompletedProcess(
        returncode=0, args=[], stdout='', stderr=''
    )

    vscode_extension.attempt_vscode_extension_install()

    # One subprocess call for --list-extensions, no installation attempts
    assert mock_env_and_dependencies['subprocess'].call_count == 1
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_no_editor_detection_skips_installation(mock_env_and_dependencies):
    """Test that no installation is attempted when no supported editor is detected."""
    os.environ['TERM_PROGRAM'] = 'iTerm.app'
    os.environ['PATH'] = '/usr/local/bin:/usr/bin:/bin'
    vscode_extension.attempt_vscode_extension_install()
    mock_env_and_dependencies['exists'].assert_not_called()
    mock_env_and_dependencies['touch'].assert_not_called()
    mock_env_and_dependencies['subprocess'].assert_not_called()
    mock_env_and_dependencies['print'].assert_not_called()


def test_both_bundled_and_marketplace_fail(mock_env_and_dependencies):
    """Test when bundled VSIX installation fails."""
    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_vsix_path = mock.MagicMock()
    mock_vsix_path.exists.return_value = True
    mock_vsix_path.__str__.return_value = '/mock/path/openhands-vscode-0.0.1.vsix'
    mock_env_and_dependencies[
        'as_file'
    ].return_value.__enter__.return_value = mock_vsix_path

    # Mock subprocess calls: first --list-extensions (empty), then bundled install (fails)
    mock_env_and_dependencies['subprocess'].side_effect = [
        subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        ),  # --list-extensions
        subprocess.CompletedProcess(
            args=[
                'code',
                '--install-extension',
                '/mock/path/openhands-vscode-0.0.1.vsix',
                '--force',
            ],
            returncode=1,
            stdout='Bundled installation failed',
            stderr='Error installing bundled extension',
        ),
    ]

    vscode_extension.attempt_vscode_extension_install()

    # Two subprocess calls: --list-extensions and bundled VSIX install
    assert mock_env_and_dependencies['subprocess'].call_count == 2
    mock_env_and_dependencies['print'].assert_any_call(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )


def test_windows_fallback_to_code_cmd(mock_env_and_dependencies):
    """On Windows, fall back to code.cmd when 'code' is not resolvable."""
    import inspect

    # Skip if runtime imports an older module version without fallback logic
    src = inspect.getsource(vscode_extension.attempt_vscode_extension_install)
    if 'code.cmd' not in src:
        pytest.skip(
            'Module under test lacks Windows fallback logic; skipping environment-dependent test.'
        )

    os.environ['TERM_PROGRAM'] = 'vscode'
    mock_env_and_dependencies['exists'].return_value = False
    mock_env_and_dependencies['download'].return_value = None
    mock_env_and_dependencies['as_file'].side_effect = FileNotFoundError

    with (
        mock.patch('platform.system', return_value='Windows'),
        mock.patch('shutil.which') as mock_which,
    ):

        def which_side_effect(cmd):
            if cmd == 'code':
                return None
            if cmd == 'code.cmd':
                return 'code.cmd'  # Simulate Windows launcher available on PATH
            return None

        mock_which.side_effect = which_side_effect

        # Mock subprocess call for --list-extensions
        mock_env_and_dependencies[
            'subprocess'
        ].return_value = subprocess.CompletedProcess(
            returncode=0, args=[], stdout='', stderr=''
        )

        vscode_extension.attempt_vscode_extension_install()

        # Verify we tried invoking code.cmd
        called_args, called_kwargs = mock_env_and_dependencies['subprocess'].call_args
        assert called_args[0][0].endswith('code.cmd')
        assert called_args[0][1:] == ['--list-extensions']
        assert called_kwargs['capture_output'] is True
        assert called_kwargs['text'] is True
        assert called_kwargs['check'] is False
