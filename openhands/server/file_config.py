# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import os
import re

from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.server.shared import config as shared_config

FILES_TO_IGNORE = [
    '.git/',
    '.DS_Store',
    'node_modules/',
    '__pycache__/',
    'lost+found/',
    '.vscode/',
]


def sanitize_filename(filename: str) -> str:
    """Sanitize the filename to prevent directory traversal"""
    # Remove any directory components
    filename = os.path.basename(filename)
    # Remove any non-alphanumeric characters except for .-_
    filename = re.sub(r'[^\w\-_\.]', '', filename)
    # Limit the filename length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[: max_length - len(ext)] + ext
    return filename


def load_file_upload_config(
    config: OpenHandsConfig = shared_config,
) -> tuple[int, bool, list[str]]:
    """Load file upload configuration from the config object.

    This function retrieves the file upload settings from the global config object.
    It handles the following settings:
    - Maximum file size for uploads
    - Whether to restrict file types
    - List of allowed file extensions

    It also performs sanity checks on the values to ensure they are valid and safe.

    Returns:
        tuple: A tuple containing:
            - max_file_size_mb (int): Maximum file size in MB. 0 means no limit.
            - restrict_file_types (bool): Whether file type restrictions are enabled.
            - allowed_extensions (set): Set of allowed file extensions.
    """
    # Retrieve values from config
    max_file_size_mb = config.file_uploads_max_file_size_mb
    restrict_file_types = config.file_uploads_restrict_file_types
    allowed_extensions = config.file_uploads_allowed_extensions

    # Sanity check for max_file_size_mb
    if not isinstance(max_file_size_mb, int) or max_file_size_mb < 0:
        logger.warning(
            f'Invalid max_file_size_mb: {max_file_size_mb}. Setting to 0 (no limit).'
        )
        max_file_size_mb = 0

    # Sanity check for allowed_extensions
    if not isinstance(allowed_extensions, (list, set)) or not allowed_extensions:
        logger.warning(
            f'Invalid allowed_extensions: {allowed_extensions}. Setting to [".*"].'
        )
        allowed_extensions = ['.*']
    else:
        # Ensure all extensions start with a dot and are lowercase
        allowed_extensions = [
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in allowed_extensions
        ]

    # If restrictions are disabled, allow all
    if not restrict_file_types:
        allowed_extensions = ['.*']

    logger.debug(
        f'File upload config: max_size={max_file_size_mb}MB, '
        f'restrict_types={restrict_file_types}, '
        f'allowed_extensions={allowed_extensions}'
    )

    return max_file_size_mb, restrict_file_types, allowed_extensions


# Load configuration
MAX_FILE_SIZE_MB, RESTRICT_FILE_TYPES, ALLOWED_EXTENSIONS = load_file_upload_config()


def is_extension_allowed(filename: str) -> bool:
    """Check if the file extension is allowed based on the current configuration.

    This function supports wildcards and files without extensions.
    The check is case-insensitive for extensions.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    if not RESTRICT_FILE_TYPES:
        return True

    file_ext = os.path.splitext(filename)[1].lower()  # Convert to lowercase
    return (
        '.*' in ALLOWED_EXTENSIONS
        or file_ext in (ext.lower() for ext in ALLOWED_EXTENSIONS)
        or (file_ext == '' and '.' in ALLOWED_EXTENSIONS)
    )


def get_unique_filename(filename: str, folder_path: str) -> str:
    """Returns unique filename on given folder_path. By checking if the given
     filename exists. If it doesn't, filename is simply returned.
     Otherwise, it append copy(#number) until the filename is unique.

    Args:
        filename (str): The name of the file to check.
        folder_path (str): directory path in which file name check is performed.

    Returns:
        string: unique filename.
    """
    name, ext = os.path.splitext(filename)
    filename_candidate = filename
    copy_index = 0

    while os.path.exists(os.path.join(folder_path, filename_candidate)):
        if copy_index == 0:
            filename_candidate = f'{name} copy{ext}'
        else:
            filename_candidate = f'{name} copy({copy_index}){ext}'
        copy_index += 1

    return filename_candidate
