# Atlas Cloud (OpenAI-compatible LLM backend)

<p align="center">
  <a href="https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=OpenHands">
    <img src="./atlas-cloud-logo.png" alt="Atlas Cloud" width="200">
  </a>
</p>

> 🎁 **[Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=OpenHands)** is a full-modal, OpenAI-compatible AI inference platform. Because OpenHands routes every LLM call through [LiteLLM](https://docs.litellm.ai/), you can use Atlas as a drop-in **bring-your-own-model** backend — a single API key gives the agent access to DeepSeek, Qwen, GLM, Kimi, MiniMax and more, with no extra provider code.
>
> Budget-friendly for always-on agents: [coding plan](https://www.atlascloud.ai/console/coding-plan).

## Why this works without a custom provider

OpenHands talks to models through LiteLLM. Any OpenAI-compatible endpoint is reachable by:

1. prefixing the model with `openai/` so LiteLLM uses its OpenAI-compatible transport, and
2. pointing `base_url` at the provider.

Atlas Cloud exposes a standard `POST /v1/chat/completions` endpoint at `https://api.atlascloud.ai/v1`, so it slots straight in.

## Configuration

### Via `config.toml`

```toml
[llm.atlas]
model = "openai/deepseek-ai/deepseek-v4-pro"
base_url = "https://api.atlascloud.ai/v1"
api_key = "<your-atlascloud-api-key>"
# deepseek-v4-pro is a reasoning model — give it enough room for the
# chain-of-thought, otherwise content can come back empty.
max_output_tokens = 4096
```

Then reference the named config (for example `llm_config = "atlas"`) or set it as the default LLM in the UI / settings.

### Via the UI (Bring your own model)

In **Settings → LLM**, pick the *advanced / custom* option and fill in:

- **Model**: `openai/deepseek-ai/deepseek-v4-pro`
- **Base URL**: `https://api.atlascloud.ai/v1`
- **API Key**: your Atlas Cloud key

> `deepseek-ai/deepseek-v4-pro` is a reasoning model — make sure the max output tokens are generous (>= 512, ideally a few thousand for agentic loops) so the answer isn't truncated by the chain-of-thought.

Never commit a real API key — keep it in `config.toml` (gitignored) or an environment variable.

## Available chat models

Official Atlas Cloud LLM list (59 models, synced with `atlascloud.ai/models`). Use them with the `openai/<model>` prefix shown above.

<details>
<summary>All Atlas Cloud chat models (59)</summary>

- **Anthropic (Claude)**: `anthropic/claude-haiku-4.5-20251001`, `anthropic/claude-opus-4.8`, `anthropic/claude-sonnet-4.6`
- **OpenAI (GPT)**: `openai/gpt-5.4`, `openai/gpt-5.5`
- **Google (Gemini)**: `google/gemini-3.1-flash-lite`, `google/gemini-3.1-pro-preview`, `google/gemini-3.5-flash`
- **Alibaba (Qwen)**: `qwen/qwen2.5-7b-instruct`, `Qwen/Qwen3-235B-A22B-Instruct-2507`, `qwen/qwen3-235b-a22b-thinking-2507`, `qwen/qwen3-30b-a3b`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `qwen/qwen3-30b-a3b-thinking-2507`, `qwen/qwen3-32b`, `qwen/qwen3-8b`, `Qwen/Qwen3-Coder`, `qwen/qwen3-coder-next`, `qwen/qwen3-max-2026-01-23`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-Next-80B-A3B-Thinking`, `Qwen/Qwen3-VL-235B-A22B-Instruct`, `qwen/qwen3-vl-235b-a22b-thinking`, `qwen/qwen3-vl-30b-a3b-instruct`, `qwen/qwen3-vl-30b-a3b-thinking`, `qwen/qwen3-vl-8b-instruct`, `qwen/qwen3.5-122b-a10b`, `qwen/qwen3.5-27b`, `qwen/qwen3.5-35b-a3b`, `qwen/qwen3.5-397b-a17b`, `qwen/qwen3.6-35b-a3b`, `qwen/qwen3.6-plus`
- **DeepSeek**: `deepseek-ai/deepseek-ocr`, `deepseek-ai/deepseek-r1-0528`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V3.1-Terminus`, `deepseek-ai/deepseek-v3.2`, `deepseek-ai/DeepSeek-V3.2-Exp`, `deepseek-ai/deepseek-v4-flash`, `deepseek-ai/deepseek-v4-pro`
- **Moonshot (Kimi)**: `moonshotai/Kimi-K2-Instruct`, `moonshotai/Kimi-K2-Instruct-0905`, `moonshotai/Kimi-K2-Thinking`, `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`
- **Zhipu (GLM)**: `zai-org/GLM-4.6`, `zai-org/glm-4.7`, `zai-org/glm-5`, `zai-org/glm-5-turbo`, `zai-org/glm-5.1`, `zai-org/glm-5v-turbo`
- **MiniMax**: `MiniMaxAI/MiniMax-M2`, `minimaxai/minimax-m2.1`, `minimaxai/minimax-m2.5`, `minimaxai/minimax-m2.7`
- **xAI**: `xai/grok-4.3`
- **Kuaishou (KAT)**: `kwaipilot/kat-coder-pro-v2`
- **Other**: `owl`

</details>

For the always-current catalogue see [atlascloud.ai/models](https://www.atlascloud.ai/models).
