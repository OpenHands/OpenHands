"""Composition of plugin marketplaces across instance/org/user scopes.

Marketplaces compose additively with precedence ``Instance < Org < User``.
Identity is the marketplace **name** (the stable, user-facing identifier and the
key used by ``plugin-name@marketplace-name`` resolution). Narrower scopes cannot
override broader ones:

- Org entries may override an instance entry of the *same name* (e.g. to flip
  ``auto_load``); a new org name is simply added.
- A user entry whose name already exists at the instance/org level is dropped
  (users add only). A new user name becomes a personal marketplace.

This module is the single source of truth for composition. It is used both by the
settings API (to render inherited/personal for the UI) and by conversation start
(to decide which marketplaces to hand to the agent-server for plugin loading).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from openhands.app_server.settings.settings_models import (
    MarketplaceRegistration,
    MarketplaceScope,
)

logger = logging.getLogger(__name__)

INSTANCE_MARKETPLACES_ENV = 'INSTANCE_DEFAULT_MARKETPLACES'
MARKETPLACE_LOADING_ENV = 'ENABLE_MARKETPLACE_PLUGIN_LOADING'

_RawMarketplace = dict[str, Any] | MarketplaceRegistration


def marketplace_plugin_loading_enabled() -> bool:
    """Whether composed marketplaces are sent to the agent-server at conversation
    start so their ``auto_load`` plugins are loaded.

    Enabled by default: the agent-server ``/api/skills`` support for
    ``registered_marketplaces`` ships in the pinned SDK (>=1.30.0). Set
    ``ENABLE_MARKETPLACE_PLUGIN_LOADING`` to a falsy value (``0``/``false``/
    ``no``/``off``) to disable the loading path as a kill switch. Storage and the
    settings UI work regardless of this flag.
    """
    raw = os.environ.get(MARKETPLACE_LOADING_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


@dataclass
class ComposedMarketplaces:
    """Result of composing marketplaces across scopes.

    ``inherited`` (instance + org) is read-only for users; ``personal`` is the
    user's own set. ``all`` is the combined set used for plugin loading.
    """

    inherited: list[MarketplaceRegistration] = field(default_factory=list)
    personal: list[MarketplaceRegistration] = field(default_factory=list)

    @property
    def all(self) -> list[MarketplaceRegistration]:
        return [*self.inherited, *self.personal]


def _derive_name_from_source(source: str) -> str:
    """Best-effort marketplace name from a source string.

    ``github:owner/repo`` / ``.../owner/repo`` -> ``repo``. Falls back to the
    raw source when no separator is present.
    """
    source = (source or '').strip()
    if ':' in source:
        source = source.split(':', 1)[-1]
    if '/' in source:
        return source.rstrip('/').split('/')[-1]
    return source


def _parse_instance_marketplaces_env(env_value: str) -> list[dict[str, Any]]:
    """Parse the raw ``INSTANCE_DEFAULT_MARKETPLACES`` value into raw dicts.

    Two formats are supported. JSON is tried on the *whole* value first so that
    arrays/objects containing commas parse correctly:

    - JSON: ``{"source": ...}`` or ``[{"source": ...}, ...]``
    - Comma-separated ``#``-delimited entries: ``source#name#ref#repo_path``

    Raises on malformed JSON; callers degrade to an empty list.
    """
    if env_value[0] in '[{':
        data = json.loads(env_value)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
        raise ValueError('JSON must be an object or a list of objects')

    entries: list[dict[str, Any]] = []
    for definition in env_value.split(','):
        definition = definition.strip()
        if not definition:
            continue
        parts = definition.split('#')
        entry: dict[str, Any] = {'source': parts[0].strip()}
        if len(parts) > 1 and parts[1].strip():
            entry['name'] = parts[1].strip()
        if len(parts) > 2 and parts[2].strip():
            entry['ref'] = parts[2].strip()
        if len(parts) > 3 and parts[3].strip():
            entry['repo_path'] = parts[3].strip()
        entries.append(entry)
    return entries


def get_instance_default_marketplaces() -> list[dict[str, Any]]:
    """Instance-level default marketplaces from the environment.

    Instance defaults auto-load by default. Parsing/validation never raises: a
    bad value is logged and yields an empty list so a single operator typo can
    never break settings loading for every user.
    """
    env_value = os.environ.get(INSTANCE_MARKETPLACES_ENV, '').strip()
    if not env_value:
        return []

    try:
        raw_entries = _parse_instance_marketplaces_env(env_value)
    except Exception as e:  # noqa: BLE001 - degrade gracefully on any bad config
        logger.warning(
            'Failed to parse %s; ignoring instance defaults: %s',
            INSTANCE_MARKETPLACES_ENV,
            e,
        )
        return []

    validated: list[dict[str, Any]] = []
    for raw in raw_entries:
        entry = dict(raw)
        entry.setdefault('auto_load', True)
        if not entry.get('name'):
            entry['name'] = _derive_name_from_source(entry.get('source', ''))
        try:
            reg = MarketplaceRegistration.model_validate(entry)
        except (ValidationError, ValueError) as e:
            logger.warning(
                'Invalid marketplace in %s: %s', INSTANCE_MARKETPLACES_ENV, e
            )
            continue
        validated.append(reg.model_dump())
    return validated


def _name_of(mp: _RawMarketplace) -> str:
    name = mp.get('name') if isinstance(mp, dict) else getattr(mp, 'name', None)
    return (name or '').strip()


def duplicate_marketplace_names(
    marketplaces: Sequence[_RawMarketplace] | None,
    reserved_names: Iterable[str] = (),
) -> list[str]:
    """Names duplicated within ``marketplaces`` or already in ``reserved_names``.

    ``name`` is the marketplace identity (used by ``plugin@marketplace``
    resolution), so writes must keep it unique. For a personal write,
    ``reserved_names`` are the inherited instance/org names the user cannot
    shadow. Returned in first-seen order.
    """
    reserved = {n.strip() for n in reserved_names if n and n.strip()}
    seen: set[str] = set()
    conflicts: list[str] = []
    for mp in marketplaces or []:
        name = _name_of(mp)
        if not name:
            continue
        if (name in reserved or name in seen) and name not in conflicts:
            conflicts.append(name)
        seen.add(name)
    return conflicts


def _coerce(raw: _RawMarketplace) -> MarketplaceRegistration | None:
    """Validate a raw entry into a model, or None (logged) if invalid."""
    try:
        if isinstance(raw, MarketplaceRegistration):
            return raw
        return MarketplaceRegistration.model_validate(raw)
    except (ValidationError, ValueError) as e:
        logger.warning('Skipping invalid marketplace entry: %s', e)
        return None


def compose_marketplaces(
    instance_marketplaces: Sequence[_RawMarketplace] | None,
    org_marketplaces: Sequence[_RawMarketplace] | None,
    user_marketplaces: Sequence[_RawMarketplace] | None,
) -> ComposedMarketplaces:
    """Compose marketplaces from the three scopes, keyed on ``name``.

    Precedence is Instance < Org < User for identity; org overrides an instance
    entry of the same name, and a user entry is dropped when its name already
    exists at a broader scope. Invalid entries are skipped. Order is preserved
    (instance, then org additions, then user additions).
    """

    def _stamp(
        reg: MarketplaceRegistration, scope: MarketplaceScope
    ) -> MarketplaceRegistration:
        return reg.model_copy(update={'scope': scope})

    # name -> registration; dict preserves insertion order and dedupes by name.
    inherited: dict[str, MarketplaceRegistration] = {}
    for raw in instance_marketplaces or []:
        reg = _coerce(raw)
        if reg is not None:
            inherited[reg.name] = _stamp(reg, MarketplaceScope.INSTANCE)
    for raw in org_marketplaces or []:
        reg = _coerce(raw)
        if reg is not None:
            # Same name as an instance entry -> org override; new name -> add.
            inherited[reg.name] = _stamp(reg, MarketplaceScope.ORG)

    personal: dict[str, MarketplaceRegistration] = {}
    for raw in user_marketplaces or []:
        reg = _coerce(raw)
        if reg is None:
            continue
        if reg.name in inherited:
            logger.debug(
                "User marketplace '%s' shadows an inherited one; ignoring", reg.name
            )
            continue
        personal[reg.name] = _stamp(reg, MarketplaceScope.PERSONAL)

    return ComposedMarketplaces(
        inherited=list(inherited.values()),
        personal=list(personal.values()),
    )


async def load_composed_marketplaces(
    user_id: str | None,
    user_marketplaces: Sequence[_RawMarketplace] | None,
    settings_store: Any,
) -> ComposedMarketplaces:
    """Gather instance + org + user marketplaces and compose them.

    ``settings_store`` must expose ``get_org_marketplaces(user_id)`` (all
    ``SettingsStore`` implementations do; OSS returns ``[]``).
    """
    instance = get_instance_default_marketplaces()
    try:
        org = await settings_store.get_org_marketplaces(user_id)
    except Exception as e:  # noqa: BLE001 - org lookup must never break composition
        logger.warning('Failed to load org marketplaces: %s', e)
        org = []
    return compose_marketplaces(instance, org, user_marketplaces)
