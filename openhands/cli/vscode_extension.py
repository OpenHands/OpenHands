import importlib.resources
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import urllib.request
from urllib.error import URLError

from openhands.core.logger import openhands_logger as logger


def download_latest_vsix_from_github() -> str | None:
    """Download latest .vsix from GitHub releases.

    Returns:
        Path to downloaded .vsix file, or None if failed
    """
    api_url = 'https://api.github.com/repos/OpenHands/OpenHands/releases'
    try:
        with urllib.request.urlopen(api_url, timeout=10) as response:
            if response.status != 200:
                logger.debug(
                    f'GitHub API request failed with status: {response.status}'
                )
                return None
            releases = json.loads(response.read().decode())
            # The GitHub API returns releases in reverse chronological order (newest first).
            # We iterate through them and use the first one that matches our extension prefix.
            for release in releases:
                if release.get('tag_name', '').startswith('ext-v'):
                    for asset in release.get('assets', []):
                        if asset.get('name', '').endswith('.vsix'):
                            download_url = asset.get('browser_download_url')
                            if not download_url:
                                continue
                            with urllib.request.urlopen(
                                download_url, timeout=30
                            ) as download_response:
                                if download_response.status != 200:
                                    logger.debug(
                                        f'Failed to download .vsix with status: {download_response.status}'
                                    )
                                    continue
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix='.vsix'
                                ) as tmp_file:
                                    tmp_file.write(download_response.read())
                                    return tmp_file.name
                    # Found the latest extension release but no .vsix asset
                    return None
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.debug(f'Failed to download from GitHub releases: {e}')
        return None
    return None


def attempt_vscode_extension_install():
    """Checks if running in a supported editor and attempts to install the OpenHands companion extension.
    This is a best-effort, one-time attempt.
    """
    # 1. Check if we are in a supported editor environment
    is_vscode_like = os.environ.get('TERM_PROGRAM') == 'vscode'
    is_windsurf = (
        os.environ.get('__CFBundleIdentifier') == 'com.exafunction.windsurf'
        or 'windsurf' in os.environ.get('PATH', '').lower()
        or any(
            'windsurf' in val.lower()
            for val in os.environ.values()
            if isinstance(val, str)
        )
    )
    if not (is_vscode_like or is_windsurf):
        return

    # 2. Determine editor-specific commands and flags
    if is_windsurf:
        editor_command, editor_name, flag_suffix = 'surf', 'Windsurf', 'windsurf'
    else:
        editor_command, editor_name, flag_suffix = 'code', 'VS Code', 'vscode'

    # 2.1 Resolve the actual editor command across platforms. Try PATH first,
    # then common Windows wrappers if needed.
    resolved = shutil.which(editor_command)
    if resolved:
        editor_command = resolved
    else:
        if editor_command == 'code':
            for candidate in ('code.cmd', 'code.exe'):
                path = shutil.which(candidate)
                if path:
                    editor_command = path
                    break
        elif editor_command == 'surf':
            for candidate in ('surf.cmd', 'surf.exe'):
                path = shutil.which(candidate)
                if path:
                    editor_command = path
                    break

    # 3. Check if we've already successfully installed the extension.
    flag_dir = pathlib.Path.home() / '.openhands'
    flag_file = flag_dir / f'.{flag_suffix}_extension_installed'
    extension_id = 'openhands.openhands-vscode'

    try:
        flag_dir.mkdir(parents=True, exist_ok=True)
        if flag_file.exists():
            return  # Already successfully installed, exit.
    except OSError as e:
        logger.debug(
            f'Could not create or check {editor_name} extension flag directory: {e}'
        )
        return  # Don't proceed if we can't manage the flag.

    # 4. Check if the extension is already installed (even without our flag).
    if _is_extension_installed(editor_command, extension_id):
        print(f'INFO: OpenHands {editor_name} extension is already installed.')
        # Create flag to avoid future checks
        _mark_installation_successful(flag_file, editor_name)
        return

    # 5. Extension is not installed, attempt installation.
    print(
        f'INFO: First-time setup: attempting to install the OpenHands {editor_name} extension...'
    )

    # Attempt 1: Install from bundled .vsix
    if _attempt_bundled_install(editor_command, editor_name):
        _mark_installation_successful(flag_file, editor_name)
        return  # Success! We are done.

    # Attempt 2: Download from GitHub Releases
    if _attempt_github_install(editor_command, editor_name):
        _mark_installation_successful(flag_file, editor_name)
        return  # Success! We are done.

    # TODO: Attempt 3: Install from Marketplace (when extension is published)
    # if _attempt_marketplace_install(editor_command, editor_name, extension_id):
    #     _mark_installation_successful(flag_file, editor_name)
    #     return  # Success! We are done.

    # If all attempts failed, inform the user (but don't create flag - allow retry).
    print(
        'INFO: Automatic installation failed. Please check the OpenHands documentation for manual installation instructions.'
    )
    print(
        f'INFO: Will retry installation next time you run OpenHands in {editor_name}.'
    )


def _mark_installation_successful(flag_file: pathlib.Path, editor_name: str) -> None:
    """Mark the extension installation as successful by creating the flag file.

    Args:
        flag_file: Path to the flag file to create
        editor_name: Human-readable name of the editor for logging
    """
    try:
        flag_file.touch()
        logger.debug(f'{editor_name} extension installation marked as successful.')
    except OSError as e:
        logger.debug(f'Could not create {editor_name} extension success flag file: {e}')


def _is_extension_installed(editor_command: str, extension_id: str) -> bool:
    """Check if the OpenHands extension is already installed.

    Args:
        editor_command: The command to run the editor (e.g., 'code', 'windsurf')
        extension_id: The extension ID to check for

    Returns:
        bool: True if extension is already installed, False otherwise
    """
    try:
        process = subprocess.run(
            [editor_command, '--list-extensions'],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode == 0:
            installed_extensions = process.stdout.strip().split('\n')
            return extension_id in installed_extensions
    except Exception as e:
        logger.debug(f'Could not check installed extensions: {e}')

    return False


def _attempt_github_install(editor_command: str, editor_name: str) -> bool:
    """Attempt to install the extension from GitHub Releases.

    Downloads the latest VSIX file from GitHub releases and attempts to install it.
    Ensures proper cleanup of temporary files.

    Args:
        editor_command: The command to run the editor (e.g., 'code', 'windsurf')
        editor_name: Human-readable name of the editor (e.g., 'VS Code', 'Windsurf')

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    vsix_path_from_github = download_latest_vsix_from_github()
    if not vsix_path_from_github:
        return False

    github_success = False
    try:
        process = subprocess.run(
            [
                editor_command,
                '--install-extension',
                vsix_path_from_github,
                '--force',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode == 0:
            print(
                f'INFO: OpenHands {editor_name} extension installed successfully from GitHub.'
            )
            github_success = True
        else:
            logger.debug(
                f'Failed to install .vsix from GitHub: {process.stderr.strip()}'
            )
    finally:
        # Clean up the downloaded file
        if os.path.exists(vsix_path_from_github):
            try:
                os.remove(vsix_path_from_github)
            except OSError as e:
                logger.debug(
                    f'Failed to delete temporary file {vsix_path_from_github}: {e}'
                )

    return github_success


def _attempt_bundled_install(editor_command: str, editor_name: str) -> bool:
    """Attempt to install the extension from the bundled VSIX file.

    Uses the VSIX file packaged with the OpenHands installation.

    Args:
        editor_command: The command to run the editor (e.g., 'code', 'windsurf')
        editor_name: Human-readable name of the editor (e.g., 'VS Code', 'Windsurf')

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        vsix_filename = 'openhands-vscode-0.0.1.vsix'
        with importlib.resources.as_file(
            importlib.resources.files('openhands').joinpath(
                'integrations', 'vscode', vsix_filename
            )
        ) as vsix_path:
            if vsix_path.exists():
                process = subprocess.run(
                    [
                        editor_command,
                        '--install-extension',
                        str(vsix_path),
                        '--force',
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if process.returncode == 0:
                    print(
                        f'INFO: Bundled {editor_name} extension installed successfully.'
                    )
                    return True
                else:
                    logger.debug(
                        f'Bundled .vsix installation failed: {process.stderr.strip()}'
                    )
            else:
                logger.debug(f'Bundled .vsix not found at {vsix_path}.')
    except Exception as e:
        logger.warning(
            f'Could not auto-install extension. Please make sure "code" command is in PATH. Error: {e}'
        )

    return False


def _attempt_marketplace_install(
    editor_command: str, editor_name: str, extension_id: str
) -> bool:
    """Attempt to install the extension from the marketplace.

    This method is currently unused as the OpenHands extension is not yet published
    to the VS Code/Windsurf marketplace. It's kept here for future use when the
    extension becomes available.

    Args:
        editor_command: The command to use ('code' or 'surf')
        editor_name: Human-readable editor name ('VS Code' or 'Windsurf')
        extension_id: The extension ID to install

    Returns:
        True if installation succeeded, False otherwise
    """
    try:
        process = subprocess.run(
            [editor_command, '--install-extension', extension_id, '--force'],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode == 0:
            print(
                f'INFO: {editor_name} extension installed successfully from the Marketplace.'
            )
            return True
        else:
            logger.debug(f'Marketplace installation failed: {process.stderr.strip()}')
            return False
    except FileNotFoundError:
        print(
            f"INFO: To complete {editor_name} integration, please ensure the '{editor_command}' command-line tool is in your PATH."
        )
        return False
    except Exception as e:
        logger.debug(
            f'An unexpected error occurred trying to install from the Marketplace: {e}'
        )
        return False
