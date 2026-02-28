# -*- coding: utf-8 -*-

# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os
import re
from shutil import rmtree


def remove(path: str) -> None:
    if os.path.exists(path):
        if os.path.isdir(path):
            rmtree(path)
        else:
            os.remove(path)


# find all indices of a list of strings that match a regex
def findall_regex(items: list[str], regex: re.Pattern[str]) -> list[int]:
    found = list()
    for i in range(0, len(items)):
        k = regex.match(items[i])
        if k:
            found.append(i)
            k = None

    return found


def split_by_regex(items: list[str], regex: re.Pattern[str]) -> list[list[str]]:
    splits = list()
    indices = findall_regex(items, regex)
    if not indices:
        splits.append(items)
        return splits

    # Add first chunk before first match
    splits.append(items[0 : indices[0]])

    # Add chunks between matches
    for i in range(len(indices) - 1):
        splits.append(items[indices[i] : indices[i + 1]])

    # Add final chunk after last match
    splits.append(items[indices[-1] :])

    return splits


# http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
def which(program: str) -> str | None:
    def is_exe(fpath: str) -> bool:
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
