# Configuring LLM providers

Agent Canvas uses LLM profiles to connect to model providers. Any
OpenAI-compatible provider can be configured by selecting a custom model and
setting its base URL and API key.

## PZERO

[PZERO](https://pzero.studio/agents) is a prepaid, OpenAI-compatible inference
marketplace. Create an account and obtain a `pzero_` API key from the PZERO
dashboard before configuring Agent Canvas.

### Configure in Agent Canvas

1. Open **Settings → LLM** and select the **Advanced** configuration.
2. Enter the following values:

   | Field        | Value                         |
   | ------------ | ----------------------------- |
   | Custom Model | `openai/deepseek-v4-flash`    |
   | Base URL     | `https://api.pzero.studio/v1` |
   | API Key      | Your `pzero_` API key         |

The `openai/` model prefix tells LiteLLM to use its OpenAI-compatible client.
The base URL must end at `/v1`; do not include `/chat/completions`.

### Configure with `config.toml`

```toml
[llm]
model = "openai/deepseek-v4-flash"
api_key = "pzero_YOUR_KEY"
base_url = "https://api.pzero.studio/v1"
```

### Verify the connection

List the model IDs currently available from PZERO:

```bash
curl -sS "https://api.pzero.studio/v1/models"
```

Send a minimal chat completion using the same model and base URL:

```bash
curl -sS -X POST "https://api.pzero.studio/v1/chat/completions" \
  -H "Authorization: Bearer $PZERO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello from PZERO"}],"stream":false}'
```
