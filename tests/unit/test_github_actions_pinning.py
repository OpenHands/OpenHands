from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATHS = (
    list((ROOT / '.github' / 'workflows').glob('*.yml'))
    + list((ROOT / '.github' / 'workflows').glob('*.yaml'))
    + list((ROOT / '.github' / 'actions').glob('**/*.yml'))
    + list((ROOT / '.github' / 'actions').glob('**/*.yaml'))
)
ALLOWLIST_PATH = ROOT / '.github' / 'allowed-mutable-actions.yml'
USES_RE = re.compile(r'uses:\s*([^\s#]+)')
PINNED_SHA_RE = re.compile(r'^[0-9a-f]{40}$')

TRUSTED_MUTABLE_OWNERS = {'actions', 'github'}
FIRST_PARTY_OWNER = 'OpenHands'


def iter_action_refs() -> list[tuple[Path, int, str, str, str]]:
    refs: list[tuple[Path, int, str, str, str]] = []
    for path in sorted(WORKFLOW_PATHS):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            match = USES_RE.search(line)
            if not match:
                continue

            value = match.group(1).strip('"\'')
            if value.startswith(('./', '../')) or '@' not in value:
                continue

            action, ref = value.rsplit('@', 1)
            refs.append((path, line_number, value, action, ref))
    return refs


def test_non_github_third_party_actions_are_pinned_to_full_commit_shas():
    mutable_refs = []

    for path, line_number, value, action, ref in iter_action_refs():
        owner = action.split('/', 1)[0]
        if owner in TRUSTED_MUTABLE_OWNERS or owner == FIRST_PARTY_OWNER:
            continue
        if PINNED_SHA_RE.fullmatch(ref):
            continue
        mutable_refs.append(f'{path.relative_to(ROOT)}:{line_number} uses {value}')

    assert mutable_refs == []


def test_first_party_mutable_refs_are_explicitly_allowlisted_with_reasons():
    allowlist = yaml.safe_load(ALLOWLIST_PATH.read_text())
    allowed_refs = {
        (entry['path'], entry['uses']): entry.get('reason', '')
        for entry in allowlist['allowed_mutable_actions']
    }
    missing_allowlist_entries = []
    missing_reasons = []

    for path, line_number, value, action, ref in iter_action_refs():
        owner = action.split('/', 1)[0]
        if owner != FIRST_PARTY_OWNER or PINNED_SHA_RE.fullmatch(ref):
            continue

        key = (str(path.relative_to(ROOT)), value)
        reason = allowed_refs.get(key)
        if reason is None:
            missing_allowlist_entries.append(
                f'{path.relative_to(ROOT)}:{line_number} uses {value}'
            )
        elif not reason.strip():
            missing_reasons.append(
                f'{path.relative_to(ROOT)}:{line_number} uses {value}'
            )

    assert missing_allowlist_entries == []
    assert missing_reasons == []
