# -*- coding: utf-8 -*-

# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os.path
import subprocess
import tempfile

from .exceptions import HunkApplyException, SubprocessException
from .patch import Change, diffobj
from .snippets import remove, which


def _apply_diff_with_subprocess(
    diff: diffobj, lines: list[str], reverse: bool = False
) -> tuple[list[str], list[str] | None]:
    # call out to patch program
    patchexec = which('patch')
    if not patchexec:
        raise SubprocessException('cannot find patch program', code=-1)

    tempdir = tempfile.gettempdir()

    filepath = os.path.join(tempdir, 'wtp-' + str(hash(diff.header)))
    oldfilepath = filepath + '.old'
    newfilepath = filepath + '.new'
    rejfilepath = filepath + '.rej'
    patchfilepath = filepath + '.patch'
    with open(oldfilepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    with open(patchfilepath, 'w') as f:
        f.write(diff.text)

    args = [
        patchexec,
        '--reverse' if reverse else '--forward',
        '--quiet',
        '--no-backup-if-mismatch',
        '-o',
        newfilepath,
        '-i',
        patchfilepath,
        '-r',
        rejfilepath,
        oldfilepath,
    ]
    ret = subprocess.call(args)

    with open(newfilepath) as f:
        lines = f.read().splitlines()

    try:
        with open(rejfilepath) as f:
            rejlines = f.read().splitlines()
    except IOError:
        rejlines = None

    remove(oldfilepath)
    remove(newfilepath)
    remove(rejfilepath)
    remove(patchfilepath)

    # do this last to ensure files get cleaned up
    if ret != 0:
        raise SubprocessException('patch program failed', code=ret)

    return lines, rejlines


def _reverse(changes: list[Change]) -> list[Change]:
    def _reverse_change(c: Change) -> Change:
        return c._replace(old=c.new, new=c.old)

    return [_reverse_change(c) for c in changes]


def apply_diff(
    diff: diffobj, text: str | list[str], reverse: bool = False, use_patch: bool = False
) -> list[str]:
    lines = text.splitlines() if isinstance(text, str) else list(text)

    if use_patch:
        lines, _ = _apply_diff_with_subprocess(diff, lines, reverse)
        return lines

    n_lines = len(lines)

    changes = _reverse(diff.changes) if reverse else diff.changes
    # check that the source text matches the context of the diff
    for old, new, line, hunk in changes:
        # might have to check for line is None here for ed scripts
        if old is not None and line is not None:
            if old > n_lines:
                raise HunkApplyException(
                    'context line {n}, "{line}" does not exist in source'.format(
                        n=old, line=line
                    ),
                    hunk=hunk,
                )
            if lines[old - 1] != line:
                # Try to normalize whitespace by replacing multiple spaces with a single space
                # This helps with patches that have different indentation levels
                normalized_line = ' '.join(line.split())
                normalized_source = ' '.join(lines[old - 1].split())
                if normalized_line != normalized_source:
                    raise HunkApplyException(
                        'context line {n}, "{line}" does not match "{sl}"'.format(
                            n=old, line=line, sl=lines[old - 1]
                        ),
                        hunk=hunk,
                    )

    # for calculating the old line
    r = 0
    i = 0

    for old, new, line, hunk in changes:
        if old is not None and new is None:
            del lines[old - 1 - r + i]
            r += 1
        elif old is None and new is not None:
            lines.insert(new - 1, line)
            i += 1
        elif old is not None and new is not None:
            # Sometimes, people remove hunks from patches, making these
            # numbers completely unreliable. Because they're jerks.
            pass

    return lines
