from pydantic import SecretStr

from openhands.utils.chatgpt_oauth_tokens import (
    decode_chatgpt_access_token_for_llm,
    encode_chatgpt_token_bundle,
    is_chatgpt_oauth_bundle,
)


def test_encode_and_detect_bundle():
    raw = encode_chatgpt_token_bundle('access', 'refresh')
    assert is_chatgpt_oauth_bundle(SecretStr(raw))
    assert not is_chatgpt_oauth_bundle(SecretStr('sk-plain'))


def test_decode_for_chatgpt_model():
    raw = encode_chatgpt_token_bundle('access', 'refresh')
    decoded = decode_chatgpt_access_token_for_llm(
        SecretStr(raw), 'chatgpt/gpt-5.2-codex'
    )
    assert decoded is not None
    assert decoded.get_secret_value() == 'access'


def test_decode_non_chatgpt_model_passthrough():
    decoded = decode_chatgpt_access_token_for_llm(
        SecretStr('plain-key'), 'openai/gpt-4o'
    )
    assert decoded is not None
    assert decoded.get_secret_value() == 'plain-key'
