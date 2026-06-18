# Pinstripes (OpenAI-compatible LLM backend)

[**Pinstripes**](https://pinstripes.io) is an OpenAI-compatible inference API. Because OpenHands routes every LLM call through [LiteLLM](https://docs.litellm.ai/), you can use Pinstripes as a drop-in **bring-your-own-model** backend — a single API key gives the agent access to DeepSeek, GLM, Qwen, MiniMax and more, with no extra provider code.

## Why this works without a custom provider

OpenHands talks to models through LiteLLM. Any OpenAI-compatible endpoint is reachable by:

1. Prefixing the model with `openai/` so LiteLLM uses its OpenAI-compatible transport, and
2. Pointing `base_url` at the provider.

Pinstripes exposes a standard `POST /v1/chat/completions` endpoint at `https://pinstripes.io/v1`, so it slots straight in.

## Setup

### 1. Get an API key

Sign up at [pinstripes.io](https://pinstripes.io) and copy your API key.

### 2. Configure OpenHands

#### Via `config.toml`

Add a named LLM block and reference it where needed:

```toml
[llm.pinstripes]
model     = "openai/ps/deepseek-v4-flash"
base_url  = "https://pinstripes.io/v1"
api_key   = "<your-pinstripes-api-key>"
```

Then reference this config by name (for example `llm_config = "pinstripes"`) or set it as the default LLM in the UI / settings.

#### Via the UI (Bring your own model)

In **Settings → LLM**, pick the *advanced / custom* option and fill in:

| Field       | Value                               |
|-------------|-------------------------------------|
| Model       | `openai/ps/deepseek-v4-flash`       |
| Base URL    | `https://pinstripes.io/v1`          |
| API Key     | your Pinstripes key                 |

Never commit a real API key — keep it in `config.toml` (gitignored) or an environment variable (`PINSTRIPES_API_KEY`).

## Available models

| Model                     | Price (per 1M tokens) |
|---------------------------|-----------------------|
| `ps/deepseek-v4-flash`    | $0.10                 |
| `ps/glm-4.5-air`          | $0.125                |
| `ps/qwen3-35b`            | $0.14                 |
| `ps/minimax-m2.7`         | $0.255                |

Use each model with the `openai/` prefix shown above (e.g. `openai/ps/qwen3-35b`).

## Further reading

- [Pinstripes documentation](https://pinstripes.io)
- [LiteLLM OpenAI-compatible endpoints](https://docs.litellm.ai/docs/providers/openai_compatible)
- [OpenHands LLM settings](https://docs.openhands.dev/openhands/usage/settings/llm-settings)
