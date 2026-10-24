# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for OpenHands CLI.

This spec file configures PyInstaller to create a standalone executable
for the OpenHands CLI application.
"""

from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    copy_metadata
)



# Get the project root directory (current working directory when running PyInstaller)
project_root = Path.cwd()

a = Analysis(
    ['openhands_cli/simple_main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include any data files that might be needed
        # Add more data files here if needed in the future
        *collect_data_files('tiktoken'),
        *collect_data_files('tiktoken_ext'),
        *collect_data_files('litellm'),
        *collect_data_files('fastmcp'),
        *collect_data_files('mcp'),
        # binaryornot loads its CSV signatures via importlib.resources.
        *collect_data_files('binaryornot'),
        # Include all data files from openhands.sdk (templates, configs, etc.)
        *collect_data_files('openhands.sdk'),
        # Include package metadata for importlib.metadata
        *copy_metadata('fastmcp'),
    ],
    hiddenimports=[
        # Explicitly include modules that might not be detected automatically
        *collect_submodules('openhands_cli'),
        *collect_submodules('prompt_toolkit'),
        # Include OpenHands SDK submodules explicitly to avoid resolution issues
        *collect_submodules('openhands.sdk'),
        *collect_submodules('openhands.tools'),
        *collect_submodules('tiktoken'),
        *collect_submodules('tiktoken_ext'),
        *collect_submodules('litellm'),
        *collect_submodules('fastmcp'),
        # Include binaryornot.data so importlib.resources can resolve it.
        *collect_submodules('binaryornot'),
        # Include mcp but exclude CLI parts that require typer
        'mcp.types',
        'mcp.client',
        'mcp.server',
        'mcp.shared',
        'openhands.tools.terminal',
        'openhands.tools.str_replace_editor',
        'openhands.tools.task_tracker',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # runtime_hooks=[str(project_root / "hooks" / "rthook_profile_imports.py")],
    excludes=[
        # Exclude unnecessary modules to reduce binary size
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        # Exclude mcp CLI parts that cause issues
        'mcp.cli',
        'prompt_toolkit.contrib.ssh',
        'fastmcp.cli',
        'boto3',
        'botocore',
        'posthog',
        'browser-use',
        'openhands.tools.browser_use'
    ],
    noarchive=False,
    # IMPORTANT: do not use optimize=2 (-OO) because it strips docstrings used by PLY/bashlex grammar
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='openhands',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip debug symbols to reduce size
    upx=True,    # Use UPX compression if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # CLI application needs console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
