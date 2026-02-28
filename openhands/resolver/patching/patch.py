# -*- coding: utf-8 -*-
# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import base64
import re
import zlib
from collections import namedtuple
from typing import Iterable

from . import exceptions
from .snippets import findall_regex, split_by_regex

header = namedtuple(
    'header',
    'index_path old_path old_version new_path new_version',
)

diffobj = namedtuple('diffobj', 'header changes text')
Change = namedtuple('Change', 'old new line hunk')

file_timestamp_str = '(.+?)(?:\t|:|  +)(.*)'
# .+? was previously [^:\t\n\r\f\v]+

# general diff regex
diffcmd_header = re.compile('^diff.* (.+) (.+)$')
unified_header_index = re.compile('^Index: (.+)$')
unified_header_old_line = re.compile(r'^--- ' + file_timestamp_str + '$')
unified_header_new_line = re.compile(r'^\+\+\+ ' + file_timestamp_str + '$')
unified_hunk_start = re.compile(r'^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)$')
unified_change = re.compile('^([-+ ])(.*)$', re.MULTILINE)

context_header_old_line = re.compile(r'^\*\*\* ' + file_timestamp_str + '$')
context_header_new_line = re.compile('^--- ' + file_timestamp_str + '$')
context_hunk_start = re.compile(r'^\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*$')
context_hunk_old = re.compile(r'^\*\*\* (\d+),?(\d*) \*\*\*\*$')
context_hunk_new = re.compile(r'^--- (\d+),?(\d*) ----$')
context_change = re.compile('^([-+ !]) (.*)$')

ed_hunk_start = re.compile(r'^(\d+),?(\d*)([acd])$')
ed_hunk_end = re.compile('^.$')
# much like forward ed, but no 'c' type
rcs_ed_hunk_start = re.compile(r'^([ad])(\d+) ?(\d*)$')

default_hunk_start = re.compile(r'^(\d+),?(\d*)([acd])(\d+),?(\d*)$')
default_hunk_mid = re.compile('^---$')
default_change = re.compile('^([><]) (.*)$')

# Headers

# git has a special index header and no end part
git_diffcmd_header = re.compile('^diff --git a/(.+) b/(.+)$')
git_header_index = re.compile(r'^index ([a-f0-9]+)..([a-f0-9]+) ?(\d*)$')
git_header_old_line = re.compile('^--- (.+)$')
git_header_new_line = re.compile(r'^\+\+\+ (.+)$')
git_header_file_mode = re.compile(r'^(new|deleted) file mode \d{6}$')
git_header_binary_file = re.compile('^Binary files (.+) and (.+) differ')
git_binary_patch_start = re.compile(r'^GIT binary patch$')
git_binary_literal_start = re.compile(r'^literal (\d+)$')
git_binary_delta_start = re.compile(r'^delta (\d+)$')
base85string = re.compile(r'^[0-9A-Za-z!#$%&()*+;<=>?@^_`{|}~-]+$')

bzr_header_index = re.compile('=== (.+)')
bzr_header_old_line = unified_header_old_line
bzr_header_new_line = unified_header_new_line

svn_header_index = unified_header_index
svn_header_timestamp_version = re.compile(r'\((?:working copy|revision (\d+))\)')
svn_header_timestamp = re.compile(r'.*(\(.*\))$')

cvs_header_index = unified_header_index
cvs_header_rcs = re.compile(r'^RCS file: (.+)(?:,\w{1}$|$)')
cvs_header_timestamp = re.compile(r'(.+)\t([\d.]+)')
cvs_header_timestamp_colon = re.compile(r':([\d.]+)\t(.+)')
old_cvs_diffcmd_header = re.compile('^diff.* (.+):(.*) (.+):(.*)$')


def parse_patch(text: str | list[str]) -> Iterable[diffobj]:
    lines = text.splitlines() if isinstance(text, str) else text

    # maybe use this to nuke all of those line endings?
    # lines = [x.splitlines()[0] for x in lines]
    lines = [x if len(x) == 0 else x.splitlines()[0] for x in lines]

    check = [
        unified_header_index,
        diffcmd_header,
        cvs_header_rcs,
        git_header_index,
        context_header_old_line,
        unified_header_old_line,
    ]

    diffs = []
    for c in check:
        diffs = split_by_regex(lines, c)
        if len(diffs) > 1:
            break

    for diff in diffs:
        difftext = '\n'.join(diff) + '\n'
        h = parse_header(diff)
        d = parse_diff(diff)
        if h or d:
            yield diffobj(header=h, changes=d, text=difftext)


def parse_header(text: str | list[str]) -> header | None:
    h = parse_scm_header(text)
    if h is None:
        h = parse_diff_header(text)
    return h


def parse_scm_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    check = [
        (git_header_index, parse_git_header),
        (old_cvs_diffcmd_header, parse_cvs_header),
        (cvs_header_rcs, parse_cvs_header),
        (svn_header_index, parse_svn_header),
    ]

    for regex, parser in check:
        diffs = findall_regex(lines, regex)
        if len(diffs) > 0:
            git_opt = findall_regex(lines, git_diffcmd_header)
            if len(git_opt) > 0:
                res = parser(lines)
                if res:
                    old_path = res.old_path
                    new_path = res.new_path
                    if old_path.startswith('a/'):
                        old_path = old_path[2:]

                    if new_path.startswith('b/'):
                        new_path = new_path[2:]

                    return header(
                        index_path=res.index_path,
                        old_path=old_path,
                        old_version=res.old_version,
                        new_path=new_path,
                        new_version=res.new_version,
                    )
            else:
                res = parser(lines)

            return res

    return None


def parse_diff_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    check = [
        (unified_header_new_line, parse_unified_header),
        (context_header_old_line, parse_context_header),
        (diffcmd_header, parse_diffcmd_header),
        # TODO:
        # git_header can handle version-less unified headers, but
        # will trim a/ and b/ in the paths if they exist...
        (git_header_new_line, parse_git_header),
    ]

    for regex, parser in check:
        diffs = findall_regex(lines, regex)
        if len(diffs) > 0:
            return parser(lines)

    return None  # no header?


def parse_diff(text: str | list[str]) -> list[Change] | None:
    if isinstance(text, str):
        lines = text.splitlines()
    else:
        lines = text

    check = [
        (unified_hunk_start, parse_unified_diff),
        (context_hunk_start, parse_context_diff),
        (default_hunk_start, parse_default_diff),
        (ed_hunk_start, parse_ed_diff),
        (rcs_ed_hunk_start, parse_rcs_ed_diff),
        (git_binary_patch_start, parse_git_binary_diff),
    ]

    for hunk, parser in check:
        diffs = findall_regex(lines, hunk)
        if len(diffs) > 0:
            return parser(lines)
    return None


def parse_git_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    old_version = None
    new_version = None
    old_path = None
    new_path = None
    cmd_old_path = None
    cmd_new_path = None
    for line in lines:
        hm = git_diffcmd_header.match(line)
        if hm:
            cmd_old_path = hm.group(1)
            cmd_new_path = hm.group(2)
            continue

        g = git_header_index.match(line)
        if g:
            old_version = g.group(1)
            new_version = g.group(2)
            continue

        # git always has its own special headers
        o = git_header_old_line.match(line)
        if o:
            old_path = o.group(1)

        n = git_header_new_line.match(line)
        if n:
            new_path = n.group(1)

        binary = git_header_binary_file.match(line)
        if binary:
            old_path = binary.group(1)
            new_path = binary.group(2)

        if old_path and new_path:
            if old_path.startswith('a/'):
                old_path = old_path[2:]

            if new_path.startswith('b/'):
                new_path = new_path[2:]
            return header(
                index_path=None,
                old_path=old_path,
                old_version=old_version,
                new_path=new_path,
                new_version=new_version,
            )

    # if we go through all of the text without finding our normal info,
    # use the cmd if available
    if cmd_old_path and cmd_new_path and old_version and new_version:
        if cmd_old_path.startswith('a/'):
            cmd_old_path = cmd_old_path[2:]

        if cmd_new_path.startswith('b/'):
            cmd_new_path = cmd_new_path[2:]

        return header(
            index_path=None,
            # wow, I kind of hate this:
            # assume /dev/null if the versions are zeroed out
            old_path='/dev/null' if old_version == '0000000' else cmd_old_path,
            old_version=old_version,
            new_path='/dev/null' if new_version == '0000000' else cmd_new_path,
            new_version=new_version,
        )

    return None


def parse_svn_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    headers = findall_regex(lines, svn_header_index)
    if len(headers) == 0:
        return None

    while len(lines) > 0:
        i = svn_header_index.match(lines[0])
        del lines[0]
        if not i:
            continue

        diff_header = parse_diff_header(lines)
        if not diff_header:
            return header(
                index_path=i.group(1),
                old_path=i.group(1),
                old_version=None,
                new_path=i.group(1),
                new_version=None,
            )

        opath = diff_header.old_path
        over = diff_header.old_version
        if over:
            oend = svn_header_timestamp_version.match(over)
            if oend and oend.group(1):
                over = int(oend.group(1))
        elif opath:
            ts = svn_header_timestamp.match(opath)
            if ts:
                opath = opath[: -len(ts.group(1))]
                oend = svn_header_timestamp_version.match(ts.group(1))
                if oend and oend.group(1):
                    over = int(oend.group(1))

        npath = diff_header.new_path
        nver = diff_header.new_version
        if nver:
            nend = svn_header_timestamp_version.match(diff_header.new_version)
            if nend and nend.group(1):
                nver = int(nend.group(1))
        elif npath:
            ts = svn_header_timestamp.match(npath)
            if ts:
                npath = npath[: -len(ts.group(1))]
                nend = svn_header_timestamp_version.match(ts.group(1))
                if nend and nend.group(1):
                    nver = int(nend.group(1))

        if not isinstance(over, int):
            over = None

        if not isinstance(nver, int):
            nver = None

        return header(
            index_path=i.group(1),
            old_path=opath,
            old_version=over,
            new_path=npath,
            new_version=nver,
        )

    return None


def parse_cvs_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    headers = findall_regex(lines, cvs_header_rcs)
    headers_old = findall_regex(lines, old_cvs_diffcmd_header)

    if headers:
        # parse rcs style headers
        while len(lines) > 0:
            i = cvs_header_index.match(lines[0])
            del lines[0]
            if not i:
                continue

            diff_header = parse_diff_header(lines)
            if diff_header:
                over = diff_header.old_version
                if over:
                    oend = cvs_header_timestamp.match(over)
                    oend_c = cvs_header_timestamp_colon.match(over)
                    if oend:
                        over = oend.group(2)
                    elif oend_c:
                        over = oend_c.group(1)

                nver = diff_header.new_version
                if nver:
                    nend = cvs_header_timestamp.match(nver)
                    nend_c = cvs_header_timestamp_colon.match(nver)
                    if nend:
                        nver = nend.group(2)
                    elif nend_c:
                        nver = nend_c.group(1)

                return header(
                    index_path=i.group(1),
                    old_path=diff_header.old_path,
                    old_version=over,
                    new_path=diff_header.new_path,
                    new_version=nver,
                )
            return header(
                index_path=i.group(1),
                old_path=i.group(1),
                old_version=None,
                new_path=i.group(1),
                new_version=None,
            )
    elif headers_old:
        # parse old style headers
        while len(lines) > 0:
            i = cvs_header_index.match(lines[0])
            del lines[0]
            if not i:
                continue

            d = old_cvs_diffcmd_header.match(lines[0])
            if not d:
                return header(
                    index_path=i.group(1),
                    old_path=i.group(1),
                    old_version=None,
                    new_path=i.group(1),
                    new_version=None,
                )

            # will get rid of the useless stuff for us
            parse_diff_header(lines)
            over = d.group(2) if d.group(2) else None
            nver = d.group(4) if d.group(4) else None
            return header(
                index_path=i.group(1),
                old_path=d.group(1),
                old_version=over,
                new_path=d.group(3),
                new_version=nver,
            )

    return None


def parse_diffcmd_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    headers = findall_regex(lines, diffcmd_header)
    if len(headers) == 0:
        return None

    while len(lines) > 0:
        d = diffcmd_header.match(lines[0])
        del lines[0]
        if d:
            return header(
                index_path=None,
                old_path=d.group(1),
                old_version=None,
                new_path=d.group(2),
                new_version=None,
            )
    return None


def parse_unified_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    headers = findall_regex(lines, unified_header_new_line)
    if len(headers) == 0:
        return None

    while len(lines) > 1:
        o = unified_header_old_line.match(lines[0])
        del lines[0]
        if o:
            n = unified_header_new_line.match(lines[0])
            del lines[0]
            if n:
                over = o.group(2)
                if len(over) == 0:
                    over = None

                nver = n.group(2)
                if len(nver) == 0:
                    nver = None

                return header(
                    index_path=None,
                    old_path=o.group(1),
                    old_version=over,
                    new_path=n.group(1),
                    new_version=nver,
                )

    return None


def parse_context_header(text: str | list[str]) -> header | None:
    lines = text.splitlines() if isinstance(text, str) else text

    headers = findall_regex(lines, context_header_old_line)
    if len(headers) == 0:
        return None

    while len(lines) > 1:
        o = context_header_old_line.match(lines[0])
        del lines[0]
        if o:
            n = context_header_new_line.match(lines[0])
            del lines[0]
            if n:
                over = o.group(2)
                if len(over) == 0:
                    over = None

                nver = n.group(2)
                if len(nver) == 0:
                    nver = None

                return header(
                    index_path=None,
                    old_path=o.group(1),
                    old_version=over,
                    new_path=n.group(1),
                    new_version=nver,
                )

    return None


def parse_default_diff(text: str | list[str]) -> list[Change] | None:
    lines = text.splitlines() if isinstance(text, str) else text

    old = 0
    new = 0
    old_len = 0
    new_len = 0
    r = 0
    i = 0

    changes = list()

    hunks = split_by_regex(lines, default_hunk_start)
    for hunk_n, hunk in enumerate(hunks):
        if not len(hunk):
            continue

        r = 0
        i = 0
        while len(hunk) > 0:
            h = default_hunk_start.match(hunk[0])
            c = default_change.match(hunk[0])
            del hunk[0]
            if h:
                old = int(h.group(1))
                if len(h.group(2)) > 0:
                    old_len = int(h.group(2)) - old + 1
                else:
                    old_len = 0

                new = int(h.group(4))
                if len(h.group(5)) > 0:
                    new_len = int(h.group(5)) - new + 1
                else:
                    new_len = 0

            elif c:
                kind = c.group(1)
                line = c.group(2)

                if kind == '<' and (r != old_len or r == 0):
                    changes.append(Change(old + r, None, line, hunk_n))
                    r += 1
                elif kind == '>' and (i != new_len or i == 0):
                    changes.append(Change(None, new + i, line, hunk_n))
                    i += 1

    if len(changes) > 0:
        return changes

    return None


def parse_unified_diff(text: str | list[str]) -> list[Change] | None:
    lines = text.splitlines() if isinstance(text, str) else text

    old = 0
    new = 0
    r = 0
    i = 0
    old_len = 0
    new_len = 0

    changes = list()

    hunks = split_by_regex(lines, unified_hunk_start)
    for hunk_n, hunk in enumerate(hunks):
        # reset counters
        r = 0
        i = 0
        while len(hunk) > 0:
            h = unified_hunk_start.match(hunk[0])
            del hunk[0]
            if h:
                # The hunk header @@ -1,6 +1,6 @@ means:
                # - Start at line 1 in the old file and show 6 lines
                # - Start at line 1 in the new file and show 6 lines
                old = int(h.group(1))  # Starting line in old file
                old_len = (
                    int(h.group(2)) if len(h.group(2)) > 0 else 1
                )  # Number of lines in old file

                new = int(h.group(3))  # Starting line in new file
                new_len = (
                    int(h.group(4)) if len(h.group(4)) > 0 else 1
                )  # Number of lines in new file

                h = None
                break

        # Process each line in the hunk
        for n in hunk:
            # Each line in a unified diff starts with a space (context), + (addition), or - (deletion)
            # The first character is the kind, the rest is the line content
            kind = (
                n[0] if len(n) > 0 else ' '
            )  # Empty lines in the hunk are treated as context lines
            line = n[1:] if len(n) > 1 else ''

            # Process the line based on its kind
            if kind == '-' and (r != old_len or r == 0):
                # Line was removed from the old file
                changes.append(Change(old + r, None, line, hunk_n))
                r += 1
            elif kind == '+' and (i != new_len or i == 0):
                # Line was added in the new file
                changes.append(Change(None, new + i, line, hunk_n))
                i += 1
            elif kind == ' ':
                # Context line - exists in both old and new file
                changes.append(Change(old + r, new + i, line, hunk_n))
                r += 1
                i += 1

    if len(changes) > 0:
        return changes

    return None


def parse_context_diff(text: str | list[str]) -> list[Change] | None:
    lines = text.splitlines() if isinstance(text, str) else text

    old = 0
    new = 0
    j = 0
    k = 0

    changes = list()

    hunks = split_by_regex(lines, context_hunk_start)
    for hunk_n, hunk in enumerate(hunks):
        if not len(hunk):
            continue

        j = 0
        k = 0
        parts = split_by_regex(hunk, context_hunk_new)
        if len(parts) != 2:
            raise exceptions.ParseException('Context diff invalid', hunk_n)

        old_hunk = parts[0]
        new_hunk = parts[1]

        while len(old_hunk) > 0:
            o = context_hunk_old.match(old_hunk[0])
            del old_hunk[0]

            if not o:
                continue

            old = int(o.group(1))
            old_len = int(o.group(2)) + 1 - old
            while len(new_hunk) > 0:
                n = context_hunk_new.match(new_hunk[0])
                del new_hunk[0]

                if not n:
                    continue

                new = int(n.group(1))
                new_len = int(n.group(2)) + 1 - new
                break
            break

        # now have old and new set, can start processing?
        if len(old_hunk) > 0 and len(new_hunk) == 0:
            msg = 'Got unexpected change in removal hunk: '
            # only removes left?
            while len(old_hunk) > 0:
                c = context_change.match(old_hunk[0])
                del old_hunk[0]

                if not c:
                    continue

                kind = c.group(1)
                line = c.group(2)

                if kind == '-' and (j != old_len or j == 0):
                    changes.append(Change(old + j, None, line, hunk_n))
                    j += 1
                elif kind == ' ' and (
                    (j != old_len and k != new_len) or (j == 0 or k == 0)
                ):
                    changes.append(Change(old + j, new + k, line, hunk_n))
                    j += 1
                    k += 1
                elif kind == '+' or kind == '!':
                    raise exceptions.ParseException(msg + kind, hunk_n)

            continue

        if len(old_hunk) == 0 and len(new_hunk) > 0:
            msg = 'Got unexpected change in removal hunk: '
            # only insertions left?
            while len(new_hunk) > 0:
                c = context_change.match(new_hunk[0])
                del new_hunk[0]

                if not c:
                    continue

                kind = c.group(1)
                line = c.group(2)

                if kind == '+' and (k != new_len or k == 0):
                    changes.append(Change(None, new + k, line, hunk_n))
                    k += 1
                elif kind == ' ' and (
                    (j != old_len and k != new_len) or (j == 0 or k == 0)
                ):
                    changes.append(Change(old + j, new + k, line, hunk_n))
                    j += 1
                    k += 1
                elif kind == '-' or kind == '!':
                    raise exceptions.ParseException(msg + kind, hunk_n)
            continue

        # both
        while len(old_hunk) > 0 and len(new_hunk) > 0:
            oc = context_change.match(old_hunk[0])
            nc = context_change.match(new_hunk[0])
            okind = None
            nkind = None

            if oc:
                okind = oc.group(1)
                oline = oc.group(2)

            if nc:
                nkind = nc.group(1)
                nline = nc.group(2)

            if not (oc or nc):
                del old_hunk[0]
                del new_hunk[0]
            elif okind == ' ' and nkind == ' ' and oline == nline:
                changes.append(Change(old + j, new + k, oline, hunk_n))
                j += 1
                k += 1
                del old_hunk[0]
                del new_hunk[0]
            elif okind == '-' or okind == '!' and (j != old_len or j == 0):
                changes.append(Change(old + j, None, oline, hunk_n))
                j += 1
                del old_hunk[0]
            elif nkind == '+' or nkind == '!' and (k != new_len or k == 0):
                changes.append(Change(None, new + k, nline, hunk_n))
                k += 1
                del new_hunk[0]
            else:
                return None

    if len(changes) > 0:
        return changes

    return None


def parse_ed_diff(text: str | list[str]) -> list[Change] | None:
    lines = text.splitlines() if isinstance(text, str) else text

    old = 0
    j = 0
    k = 0

    r = 0
    i = 0

    changes = list()

    hunks = split_by_regex(lines, ed_hunk_start)
    hunks.reverse()
    for hunk_n, hunk in enumerate(hunks):
        if not len(hunk):
            continue
        j = 0
        k = 0
        while len(hunk) > 0:
            o = ed_hunk_start.match(hunk[0])
            del hunk[0]

            if not o:
                continue

            old = int(o.group(1))
            old_end = int(o.group(2)) if len(o.group(2)) else old

            hunk_kind = o.group(3)
            if hunk_kind == 'd':
                k = 0
                while old_end >= old:
                    changes.append(Change(old + k, None, None, hunk_n))
                    r += 1
                    k += 1
                    old_end -= 1
                continue

            while len(hunk) > 0:
                e = ed_hunk_end.match(hunk[0])
                if not e and hunk_kind == 'c':
                    k = 0
                    while old_end >= old:
                        changes.append(Change(old + k, None, None, hunk_n))
                        r += 1
                        k += 1
                        old_end -= 1

                    # I basically have no idea why this works
                    # for these tests.
                    changes.append(
                        Change(
                            None,
                            old - r + i + k + j,
                            hunk[0],
                            hunk_n,
                        )
                    )
                    i += 1
                    j += 1
                if not e and hunk_kind == 'a':
                    changes.append(
                        Change(
                            None,
                            old - r + i + 1,
                            hunk[0],
                            hunk_n,
                        )
                    )
                    i += 1

                del hunk[0]

    if len(changes) > 0:
        return changes

    return None


def parse_rcs_ed_diff(text: str | list[str]) -> list[Change] | None:
    # much like forward ed, but no 'c' type
    lines = text.splitlines() if isinstance(text, str) else text

    old = 0
    j = 0
    size = 0
    total_change_size = 0

    changes = list()

    hunks = split_by_regex(lines, rcs_ed_hunk_start)
    for hunk_n, hunk in enumerate(hunks):
        if len(hunk):
            j = 0
            while len(hunk) > 0:
                o = rcs_ed_hunk_start.match(hunk[0])
                del hunk[0]

                if not o:
                    continue

                hunk_kind = o.group(1)
                old = int(o.group(2))
                size = int(o.group(3)) if o.group(3) else 0

                if hunk_kind == 'a':
                    old += total_change_size + 1
                    total_change_size += size
                    while size > 0 and len(hunk) > 0:
                        changes.append(Change(None, old + j, hunk[0], hunk_n))
                        j += 1
                        size -= 1

                        del hunk[0]

                elif hunk_kind == 'd':
                    total_change_size -= size
                    while size > 0:
                        changes.append(Change(old + j, None, None, hunk_n))
                        j += 1
                        size -= 1

    if len(changes) > 0:
        return changes
    return None


def parse_git_binary_diff(text: str | list[str]) -> list[Change] | None:
    lines = text.splitlines() if isinstance(text, str) else text

    changes: list[Change] = list()

    old_version = None
    new_version = None
    cmd_old_path = None
    cmd_new_path = None
    # the sizes are used as latch-up
    new_size = 0
    old_size = 0
    old_encoded = ''
    new_encoded = ''
    for line in lines:
        if cmd_old_path is None and cmd_new_path is None:
            hm = git_diffcmd_header.match(line)
            if hm:
                cmd_old_path = hm.group(1)
                cmd_new_path = hm.group(2)
                continue

        if old_version is None and new_version is None:
            g = git_header_index.match(line)
            if g:
                old_version = g.group(1)
                new_version = g.group(2)
                continue

        # the first is added file
        if new_size == 0:
            literal = git_binary_literal_start.match(line)
            if literal:
                new_size = int(literal.group(1))
                continue
            delta = git_binary_delta_start.match(line)
            if delta:
                # not supported
                new_size = 0
                continue
        elif new_size > 0:
            if base85string.match(line):
                assert len(line) >= 6 and ((len(line) - 1) % 5) == 0
                new_encoded += line[1:]
            elif 0 == len(line):
                if new_encoded:
                    decoded = base64.b85decode(new_encoded)
                    added_data = zlib.decompress(decoded)
                    assert new_size == len(added_data)
                    change = Change(None, 0, added_data, None)
                    changes.append(change)
                new_size = 0
                new_encoded = ''
            else:
                # Invalid line format
                new_size = 0
                new_encoded = ''

        # the second is removed file
        if old_size == 0:
            literal = git_binary_literal_start.match(line)
            if literal:
                old_size = int(literal.group(1))
            delta = git_binary_delta_start.match(line)
            if delta:
                # not supported
                old_size = 0
                continue
        elif old_size > 0:
            if base85string.match(line):
                assert len(line) >= 6 and ((len(line) - 1) % 5) == 0
                old_encoded += line[1:]
            elif 0 == len(line):
                if old_encoded:
                    decoded = base64.b85decode(old_encoded)
                    removed_data = zlib.decompress(decoded)
                    assert old_size == len(removed_data)
                    change = Change(0, None, None, removed_data)
                    changes.append(change)
                old_size = 0
                old_encoded = ''
            else:
                # Invalid line format
                old_size = 0
                old_encoded = ''

    return changes
