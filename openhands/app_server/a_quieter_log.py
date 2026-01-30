"""Module to supress warnings about deprecations in transitive dependencies
which appear as errors in the logs because they are logged to stderr."""

import warnings

_is_quiet = False


def quieten_log():
    global _is_quiet
    if _is_quiet:
        return

    # Suppress deprecation warnings from dependencies before they're imported
    # aifc was removed in Python 3.13 but speech_recognition still references it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    warnings.filterwarnings('ignore', category=SyntaxWarning, module=r'pydub\.utils')

    _is_quiet = True
