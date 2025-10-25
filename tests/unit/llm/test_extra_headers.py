import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from openhands.core.config import LLMConfig
from openhands.llm.async_llm import AsyncLLM
from openhands.llm.llm import LLM
from openhands.llm.streaming_llm import StreamingLLM


@pytest.fixture
def extra_headers_env(monkeypatch):
    headers = {
        'editor-version': 'vscode/1.85.1',
        'Copilot-Integration-Id': 'vscode-chat',
    }
    monkeypatch.setenv('LLM_EXTRA_HEADERS', json.dumps(headers))
    return headers


def test_llm_passes_extra_headers_to_litellm_completion(extra_headers_env):
    cfg = LLMConfig(model='gpt-4o', api_key='test_key')

    with patch('openhands.llm.llm.litellm_completion') as mock_completion:

        def _side_effect(*args, **kwargs):
            assert 'extra_headers' in kwargs
            assert kwargs['extra_headers'] == extra_headers_env
            # minimal response structure expected by wrapper
            return {'id': 'resp-1', 'choices': [{'message': {'content': 'ok'}}]}

        mock_completion.side_effect = _side_effect
        llm = LLM(cfg, service_id='svc')
        resp = llm.completion(messages=[{'role': 'user', 'content': 'hi'}])
        assert resp['choices'][0]['message']['content'] == 'ok'


@pytest.mark.asyncio
async def test_async_llm_passes_extra_headers_to_litellm_acompletion(extra_headers_env):
    cfg = LLMConfig(model='gpt-4o', api_key='test_key')

    async def _async_side_effect(*args, **kwargs):
        assert 'extra_headers' in kwargs
        assert kwargs['extra_headers'] == extra_headers_env
        return {'id': 'resp-2', 'choices': [{'message': {'content': 'ok'}}]}

    with patch(
        'openhands.llm.async_llm.litellm_acompletion',
        new=AsyncMock(side_effect=_async_side_effect),
    ):
        llm = AsyncLLM(cfg, service_id='svc')
        resp = await llm.async_completion(
            messages=[{'role': 'user', 'content': 'hi'}], stream=False
        )
        assert resp['choices'][0]['message']['content'] == 'ok'


@pytest.mark.asyncio
async def test_streaming_llm_passes_extra_headers_to_litellm_acompletion(
    extra_headers_env,
):
    cfg = LLMConfig(model='gpt-4o', api_key='test_key')

    async def _gen():
        for chunk in [
            {'choices': [{'delta': {'content': 'hello'}}]},
            {'choices': [{'delta': {'content': ' world'}}]},
        ]:
            yield chunk
            await asyncio.sleep(0)

    async def _async_side_effect(*args, **kwargs):
        assert 'extra_headers' in kwargs
        assert kwargs['extra_headers'] == extra_headers_env
        return _gen()

    with patch(
        'openhands.llm.async_llm.litellm_acompletion',
        new=AsyncMock(side_effect=_async_side_effect),
    ):
        llm = StreamingLLM(cfg, service_id='svc')
        collected = []
        async for chunk in llm.async_streaming_completion(
            messages=[{'role': 'user', 'content': 'hi'}], stream=True
        ):
            collected.append(chunk['choices'][0]['delta']['content'])
        assert ''.join(collected) == 'hello world'
