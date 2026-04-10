"""Make LiteLLM ChatGPT provider use api_key from requests when provided (not only local file)."""

from __future__ import annotations

import base64
import json


def _extract_account_id_from_token(token: str) -> str | None:
    try:
        parts = token.split('.')
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        payload_b64 += '=' * (-len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_bytes.decode('utf-8'))
    except Exception:
        return None
    auth_claims = claims.get('https://api.openai.com/auth')
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get('chatgpt_account_id')
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


_PATCHED = False


def apply_litellm_chatgpt_api_key_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
    from litellm.llms.chatgpt.common_utils import (
        ensure_chatgpt_session_id,
        get_chatgpt_default_headers,
    )
    from litellm.llms.openai.openai import OpenAIConfig

    _orig_get = ChatGPTConfig._get_openai_compatible_provider_info
    _orig_validate = ChatGPTConfig.validate_environment

    def _patched_get(
        self: ChatGPTConfig,
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str,
    ):
        dynamic_api_base = self.authenticator.get_api_base()
        if api_key:
            return dynamic_api_base, api_key, custom_llm_provider
        return _orig_get(self, model, api_base, api_key, custom_llm_provider)

    def _patched_validate(
        self: ChatGPTConfig,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        validated_headers = OpenAIConfig.validate_environment(
            self,
            headers,
            model,
            messages,
            optional_params,
            litellm_params,
            api_key,
            api_base,
        )
        account_id: str | None = None
        if api_key:
            account_id = _extract_account_id_from_token(api_key)
        else:
            account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            api_key or '', account_id, session_id
        )
        return {**default_headers, **validated_headers}

    ChatGPTConfig._get_openai_compatible_provider_info = _patched_get
    ChatGPTConfig.validate_environment = _patched_validate
    _PATCHED = True
