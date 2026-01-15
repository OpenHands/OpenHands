"""Deprecation utilities for OpenHands legacy packages.

This module provides a mechanism to emit a single consolidated deprecation
warning for the legacy 'core', 'agenthub', and 'runtime' packages.
"""

import warnings

_LEGACY_PACKAGES_WARNING_EMITTED = False


def warn_legacy_packages_deprecated() -> None:
    """Emit a single deprecation warning for legacy packages.

    This function ensures only one warning is emitted regardless of how many
    times it's called or which deprecated package is imported first.
    """
    global _LEGACY_PACKAGES_WARNING_EMITTED
    if _LEGACY_PACKAGES_WARNING_EMITTED:
        return

    _LEGACY_PACKAGES_WARNING_EMITTED = True
    warnings.warn(
        "The 'core', 'agenthub', and 'runtime' packages are deprecated since "
        "version 1.0.0 and will be removed on April 1, 2025. Please migrate to "
        "the OpenHands SDK: https://github.com/All-Hands-AI/openhands-sdk",
        DeprecationWarning,
        stacklevel=3,
    )
