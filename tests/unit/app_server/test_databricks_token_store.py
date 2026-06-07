"""Unit tests for the server-side U2M session store (TTL + LRU eviction)."""

from __future__ import annotations

import time

from openhands.app_server.auth.databricks_token_store import U2MSessionStore


def test_put_and_get_roundtrip() -> None:
    store = U2MSessionStore()
    store.put('sid-1', {'tokens': {'access_token': 'a'}})
    assert store.get('sid-1') == {'tokens': {'access_token': 'a'}}


def test_put_merges_into_existing_record() -> None:
    store = U2MSessionStore()
    store.put('sid-1', {'tokens': {'access_token': 'a'}})
    store.put('sid-1', {'oauth_client_secret': 's'})
    record = store.get('sid-1')
    assert record == {'tokens': {'access_token': 'a'}, 'oauth_client_secret': 's'}


def test_get_returns_copy_not_live_reference() -> None:
    store = U2MSessionStore()
    store.put('sid-1', {'tokens': {'access_token': 'a'}})
    got = store.get('sid-1')
    got['tokens'] = 'mutated'
    # Internal record must be unaffected by mutation of the returned copy.
    assert store.get('sid-1') == {'tokens': {'access_token': 'a'}}


def test_get_missing_returns_none() -> None:
    store = U2MSessionStore()
    assert store.get('nope') is None
    assert store.get(None) is None
    assert store.get('') is None


def test_delete_removes_record() -> None:
    store = U2MSessionStore()
    store.put('sid-1', {'tokens': {}})
    store.delete('sid-1')
    assert store.get('sid-1') is None
    # Deleting unknown / empty ids is a no-op (must not raise).
    store.delete('unknown')
    store.delete(None)


def test_ttl_eviction() -> None:
    store = U2MSessionStore(ttl_s=0.05)
    store.put('sid-1', {'tokens': {}})
    assert store.get('sid-1') is not None
    time.sleep(0.08)
    # Expired entries are evicted on access.
    assert store.get('sid-1') is None


def test_sliding_ttl_refreshes_on_access() -> None:
    store = U2MSessionStore(ttl_s=0.12)
    store.put('sid-1', {'tokens': {}})
    time.sleep(0.08)
    assert store.get('sid-1') is not None  # refreshes expiry
    time.sleep(0.08)
    # Still alive because the previous get() slid the window forward.
    assert store.get('sid-1') is not None


def test_max_entries_evicts_least_recently_used() -> None:
    store = U2MSessionStore(max_entries=2)
    store.put('sid-1', {'v': 1})
    store.put('sid-2', {'v': 2})
    # Touch sid-1 so sid-2 becomes the LRU entry.
    store.get('sid-1')
    store.put('sid-3', {'v': 3})  # exceeds cap → evicts LRU (sid-2)
    assert store.get('sid-1') is not None
    assert store.get('sid-2') is None
    assert store.get('sid-3') is not None
