# OpenCode Go setup for this OpenHands fork

OpenCode Go should be configured through OpenHands' existing custom/OpenAI-compatible LLM settings. Keep the model and base URL configurable because provider details can change.

## Required secret

Create a Codespaces repository secret named:

```text
OPENCODE_GO_API_KEY
```

Do not commit the key to this repository.

## Codespaces environment

After creating or updating the secret, restart the Codespace and verify presence without printing the value:

```bash
if [ -n "${OPENCODE_GO_API_KEY:-}" ]; then
  echo "OpenCode Go API key is available"
else
  echo "OpenCode Go API key is missing" >&2
  exit 1
fi
```

Set the provider-specific values in the Codespace environment:

```bash
export OPENCODE_GO_MODEL="<confirmed-model-id>"
export OPENCODE_GO_BASE_URL="<confirmed-openai-compatible-base-url>"
```

## Map into OpenHands

For local configuration, map the provider values into the LLM settings expected by the current OpenHands build:

```bash
export LLM_API_KEY="$OPENCODE_GO_API_KEY"
export LLM_MODEL="openai/$OPENCODE_GO_MODEL"
export LLM_BASE_URL="$OPENCODE_GO_BASE_URL"
```

The `openai/` prefix is appropriate only when the confirmed endpoint is OpenAI-compatible and OpenHands routes it through LiteLLM.

You can also enter the same model, key, and base URL through the OpenHands LLM Settings UI instead of exporting the generic `LLM_*` variables.

## Important boundary

ChatGPT/Codex browser authentication is not a replacement for an OpenAI-compatible API key. Keep application login, ChatGPT/Codex login, OpenAI API credentials, and OpenCode Go credentials separate.

## Validation

1. Confirm the key exists without printing it.
2. Confirm the current model ID and base URL from the provider dashboard or official documentation.
3. Start OpenHands.
4. Save the custom LLM settings.
5. Run a minimal prompt and inspect server logs for authentication, model-name, or endpoint errors.
