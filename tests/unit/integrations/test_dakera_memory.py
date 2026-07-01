"""Unit tests for the Dakera memory integration.

All HTTP calls are mocked — no live Dakera server is required.
"""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openhands.app_server.integrations.dakera.config import DakeraConfig
from openhands.app_server.integrations.dakera.enricher import (
    DakeraConversationSecretEnricher,
    _build_memory_block,
    _derive_recall_query,
    _MAX_MEMORY_CONTENT_CHARS,
)
from openhands.app_server.integrations.dakera.memory_client import DakeraMemoryClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(content: str, score: float = 0.9, memory_id: str = 'mem-1') -> dict:
    return {
        'memory': {'id': memory_id, 'content': content, 'agent_id': 'openhands'},
        'score': score,
    }


def _mock_user(initial_text: str | None = None) -> MagicMock:
    """Build a minimal UserInfo-like mock."""
    user = MagicMock()
    if initial_text is not None:
        msg = MagicMock()
        msg.text = initial_text
        conv_settings = MagicMock()
        conv_settings.initial_message = msg
        user.conversation_settings = conv_settings
    else:
        user.conversation_settings = None
    return user


def _mock_context() -> MagicMock:
    return MagicMock()


def _mock_jwt_service() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# DakeraConfig
# ---------------------------------------------------------------------------


class TestDakeraConfig:
    def test_defaults(self, monkeypatch):
        for key in ('DAKERA_API_URL', 'DAKERA_API_KEY', 'DAKERA_AGENT_ID',
                    'DAKERA_TOP_K', 'DAKERA_TIMEOUT', 'DAKERA_ENABLED'):
            monkeypatch.delenv(key, raising=False)

        cfg = DakeraConfig()
        assert cfg.api_url == 'http://localhost:3300'
        assert cfg.api_key is None
        assert cfg.agent_id == 'openhands'
        assert cfg.top_k == 5
        assert cfg.timeout == 5.0
        assert cfg.enabled is True

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv('DAKERA_API_URL', 'http://dakera.example.com')
        monkeypatch.setenv('DAKERA_API_KEY', 'secret-key')
        monkeypatch.setenv('DAKERA_AGENT_ID', 'my-agent')
        monkeypatch.setenv('DAKERA_TOP_K', '10')
        monkeypatch.setenv('DAKERA_TIMEOUT', '3.0')
        monkeypatch.setenv('DAKERA_ENABLED', 'false')

        cfg = DakeraConfig()
        assert cfg.api_url == 'http://dakera.example.com'
        assert cfg.api_key == 'secret-key'
        assert cfg.agent_id == 'my-agent'
        assert cfg.top_k == 10
        assert cfg.timeout == 3.0
        assert cfg.enabled is False

    def test_constructor_kwargs_override_env(self, monkeypatch):
        monkeypatch.setenv('DAKERA_API_URL', 'http://env.example.com')
        cfg = DakeraConfig(api_url='http://kwarg.example.com', enabled=False)
        assert cfg.api_url == 'http://kwarg.example.com'
        assert cfg.enabled is False

    def test_auth_headers_with_key(self):
        cfg = DakeraConfig(api_key='tok')
        assert cfg.auth_headers == {'Authorization': 'Bearer tok'}

    def test_auth_headers_without_key(self, monkeypatch):
        monkeypatch.delenv('DAKERA_API_KEY', raising=False)
        cfg = DakeraConfig()
        assert cfg.auth_headers == {}

    def test_disabled_variants(self, monkeypatch):
        for val in ('false', 'False', '0', 'no'):
            monkeypatch.setenv('DAKERA_ENABLED', val)
            cfg = DakeraConfig()
            assert cfg.enabled is False, f'Expected disabled for DAKERA_ENABLED={val!r}'


# ---------------------------------------------------------------------------
# DakeraMemoryClient
# ---------------------------------------------------------------------------


class TestDakeraMemoryClient:
    def _client(self, api_url='http://dakera.example.com', api_key=None):
        cfg = DakeraConfig(api_url=api_url, api_key=api_key, agent_id='openhands')
        return DakeraMemoryClient(cfg)

    # ---- store ------------------------------------------------------------

    async def test_store_sends_correct_payload(self):
        client = self._client(api_key='key')
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            'memory': {'id': 'abc', 'content': 'hello', 'agent_id': 'openhands'}
        }

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value = mock_http

            result = await client.store(
                'hello',
                session_id='sess-1',
                importance=0.8,
                tags=['foo'],
                metadata={'k': 'v'},
            )

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs['json']
        assert payload['content'] == 'hello'
        assert payload['agent_id'] == 'openhands'
        assert payload['session_id'] == 'sess-1'
        assert payload['importance'] == 0.8
        assert payload['tags'] == ['foo']
        assert payload['metadata'] == {'k': 'v'}
        assert result == {'id': 'abc', 'content': 'hello', 'agent_id': 'openhands'}

    async def test_store_includes_auth_header(self):
        client = self._client(api_key='my-token')
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {'memory': {}}

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value = mock_http

            await client.store('test content')

        headers = mock_http.post.call_args.kwargs['headers']
        assert headers.get('Authorization') == 'Bearer my-token'

    async def test_store_returns_empty_on_http_error(self):
        client = self._client()

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            error_response = MagicMock()
            error_response.status_code = 500
            error_response.text = 'Internal Server Error'
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    'Server error', request=MagicMock(), response=error_response
                )
            )
            MockAsyncClient.return_value = mock_http

            result = await client.store('content')

        assert result == {}

    async def test_store_returns_empty_on_timeout(self):
        client = self._client()

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(
                side_effect=httpx.TimeoutException('timed out')
            )
            MockAsyncClient.return_value = mock_http

            result = await client.store('content')

        assert result == {}

    async def test_store_returns_empty_on_connection_error(self):
        client = self._client()

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(
                side_effect=httpx.RequestError('connection refused')
            )
            MockAsyncClient.return_value = mock_http

            result = await client.store('content')

        assert result == {}

    # ---- search -----------------------------------------------------------

    async def test_search_sends_correct_payload(self):
        client = self._client()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            'memories': [_make_hit('some memory', score=0.95)]
        }

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value = mock_http

            results = await client.search('fix the bug', top_k=3)

        payload = mock_http.post.call_args.kwargs['json']
        assert payload['agent_id'] == 'openhands'
        assert payload['query'] == 'fix the bug'
        assert payload['top_k'] == 3
        assert len(results) == 1
        assert results[0]['score'] == 0.95

    async def test_search_uses_config_top_k_by_default(self):
        cfg = DakeraConfig(top_k=7)
        client = DakeraMemoryClient(cfg)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {'memories': []}

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value = mock_http

            await client.search('query')

        payload = mock_http.post.call_args.kwargs['json']
        assert payload['top_k'] == 7

    async def test_search_returns_empty_list_on_error(self):
        client = self._client()

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            error_response = MagicMock()
            error_response.status_code = 503
            error_response.text = 'Service Unavailable'
            mock_http.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    'Error', request=MagicMock(), response=error_response
                )
            )
            MockAsyncClient.return_value = mock_http

            results = await client.search('query')

        assert results == []

    # ---- forget -----------------------------------------------------------

    async def test_forget_all_sends_agent_id_only(self):
        client = self._client()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {'deleted': 3}

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value = mock_http

            ok = await client.forget()

        payload = mock_http.post.call_args.kwargs['json']
        assert payload == {'agent_id': 'openhands'}
        assert ok is True

    async def test_forget_specific_ids(self):
        client = self._client()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {'deleted': 2}

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            MockAsyncClient.return_value = mock_http

            ok = await client.forget(['id-1', 'id-2'])

        payload = mock_http.post.call_args.kwargs['json']
        assert payload['memory_ids'] == ['id-1', 'id-2']
        assert ok is True

    async def test_forget_returns_false_on_error(self):
        client = self._client()

        with patch('httpx.AsyncClient') as MockAsyncClient:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(
                side_effect=httpx.RequestError('no route to host')
            )
            MockAsyncClient.return_value = mock_http

            ok = await client.forget()

        assert ok is False

    # ---- shared client ----------------------------------------------------

    async def test_uses_shared_http_client_when_provided(self):
        cfg = DakeraConfig()
        shared_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {'memories': []}
        shared_client.post = AsyncMock(return_value=mock_response)

        memory_client = DakeraMemoryClient(cfg, http_client=shared_client)
        await memory_client.search('test')

        shared_client.post.assert_called_once()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestBuildMemoryBlock:
    def test_empty_list_returns_empty_string(self):
        assert _build_memory_block([]) == ''

    def test_single_hit_contains_content(self):
        hits = [_make_hit('Remember to use Python 3.12')]
        block = _build_memory_block(hits)
        assert 'Remember to use Python 3.12' in block
        assert '<dakera_memory>' in block
        assert '</dakera_memory>' in block

    def test_multiple_hits_are_numbered(self):
        hits = [_make_hit('first'), _make_hit('second', memory_id='mem-2')]
        block = _build_memory_block(hits)
        assert '[1]' in block
        assert '[2]' in block

    def test_long_content_is_truncated(self):
        long_content = 'x' * (_MAX_MEMORY_CONTENT_CHARS + 100)
        hits = [_make_hit(long_content)]
        block = _build_memory_block(hits)
        # Content in the block should not exceed the limit by much
        # (the limit + the truncation suffix '…')
        assert long_content not in block
        assert '…' in block

    def test_empty_content_hits_are_skipped(self):
        hits = [{'memory': {'content': ''}, 'score': 0.9}]
        block = _build_memory_block(hits)
        assert block == ''

    def test_score_is_formatted_in_output(self):
        hits = [_make_hit('test content', score=0.75)]
        block = _build_memory_block(hits)
        assert '0.75' in block

    def test_missing_content_key_is_skipped(self):
        hits = [{'memory': {}, 'score': 0.5}]
        block = _build_memory_block(hits)
        assert block == ''


class TestDeriveRecallQuery:
    def test_uses_initial_message_text(self):
        user = _mock_user('Fix the authentication bug in the login service')
        query = _derive_recall_query(user)
        assert query == 'Fix the authentication bug in the login service'

    def test_falls_back_to_generic_query_when_no_message(self):
        user = _mock_user(initial_text=None)
        query = _derive_recall_query(user)
        assert query == 'recent tasks and context'

    def test_long_message_is_capped(self):
        long_text = 'a' * 2000
        user = _mock_user(long_text)
        query = _derive_recall_query(user)
        assert len(query) <= 1000

    def test_handles_missing_conversation_settings_gracefully(self):
        user = MagicMock()
        del user.conversation_settings  # attribute does not exist
        query = _derive_recall_query(user)
        assert query == 'recent tasks and context'


# ---------------------------------------------------------------------------
# DakeraConversationSecretEnricher
# ---------------------------------------------------------------------------


class TestDakeraConversationSecretEnricher:
    def _enricher(self, enabled=True, hits=None):
        cfg = DakeraConfig(enabled=enabled)
        mock_client = AsyncMock(spec=DakeraMemoryClient)
        mock_client.search = AsyncMock(return_value=hits or [])
        return DakeraConversationSecretEnricher(config=cfg, client=mock_client)

    async def test_disabled_returns_original_suffix(self):
        enricher = self._enricher(enabled=False)
        user = _mock_user('some task')
        result = await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix='original suffix',
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        assert result.system_message_suffix == 'original suffix'
        assert result.secrets == {}

    async def test_no_hits_returns_original_suffix(self):
        enricher = self._enricher(enabled=True, hits=[])
        user = _mock_user('debug performance issue')
        result = await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix='my suffix',
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        assert result.system_message_suffix == 'my suffix'

    async def test_hits_are_injected_into_suffix(self):
        hits = [_make_hit('Use pytest for testing', score=0.88)]
        enricher = self._enricher(enabled=True, hits=hits)
        user = _mock_user('write tests')
        result = await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix=None,
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        assert result.system_message_suffix is not None
        assert 'Use pytest for testing' in result.system_message_suffix
        assert '<dakera_memory>' in result.system_message_suffix

    async def test_memory_block_is_prepended_to_existing_suffix(self):
        hits = [_make_hit('previous context')]
        enricher = self._enricher(enabled=True, hits=hits)
        user = _mock_user('new task')
        result = await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix='original',
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        suffix = result.system_message_suffix
        assert suffix is not None
        # Memory block must come before the original suffix
        mem_pos = suffix.index('<dakera_memory>')
        orig_pos = suffix.index('original')
        assert mem_pos < orig_pos

    async def test_client_error_degrades_gracefully(self):
        cfg = DakeraConfig(enabled=True)
        mock_client = AsyncMock(spec=DakeraMemoryClient)
        mock_client.search = AsyncMock(side_effect=RuntimeError('unexpected error'))
        enricher = DakeraConversationSecretEnricher(config=cfg, client=mock_client)
        user = _mock_user('test task')
        result = await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix='fallback',
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        # Must fall back to the original suffix, not raise
        assert result.system_message_suffix == 'fallback'

    async def test_search_is_called_with_initial_message(self):
        hits = [_make_hit('result')]
        cfg = DakeraConfig(enabled=True)
        mock_client = AsyncMock(spec=DakeraMemoryClient)
        mock_client.search = AsyncMock(return_value=hits)
        enricher = DakeraConversationSecretEnricher(config=cfg, client=mock_client)
        user = _mock_user('refactor the database layer')
        await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix=None,
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert 'refactor the database layer' in call_args.args[0]

    async def test_no_secrets_are_injected(self):
        hits = [_make_hit('some memory')]
        enricher = self._enricher(enabled=True, hits=hits)
        user = _mock_user('task')
        result = await enricher.enrich(
            user_context=_mock_context(),
            user=user,
            trigger=None,
            system_message_suffix=None,
            web_url=None,
            jwt_service=_mock_jwt_service(),
            access_token_hard_timeout=None,
        )
        # Memory integration only injects context — no secrets
        assert result.secrets == {}
