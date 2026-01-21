# Claude Code CLI Transport

This document describes how to use the Claude Code CLI transport layer in OpenHands for subscription-based authentication.

## Overview

The Claude Code CLI transport allows OpenHands to use the Claude Code CLI subscription instead of direct API keys. This is useful for organizations that have Claude Code subscriptions and want to use them with OpenHands.

## Architecture

```
User Request → Agent → LLM.completion() → ClaudeCodeCLITransport → Claude Code CLI subprocess
```

Instead of making direct API calls to Anthropic, this transport spawns the Claude Code CLI as a subprocess and communicates with it.

## Prerequisites

1. **Claude Code CLI**: Install the CLI globally:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **OAuth Token**: Get a token using:
   ```bash
   claude setup-token
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_CLAUDE_CLI` | Enable Claude CLI transport | `false` |
| `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token for authentication | - |
| `CLAUDE_CLI_PATH` | Path to Claude CLI executable | `claude` |
| `CLAUDE_CLI_TIMEOUT` | Timeout in seconds | `300` |

### LLM Config Options

Add these to your OpenHands configuration:

```toml
[llm]
model = "claude-sonnet-4-20250514"
use_claude_cli = true
claude_cli_path = "claude"
claude_oauth_token = "sk-ant-..."
claude_cli_timeout = 300
```

## Docker Usage

### Build the Image

```bash
cd /path/to/openhands
docker build -t openhands-claude-cli -f containers/claude-cli/Dockerfile .
```

### Run the Container

```bash
docker run \
  -e CLAUDE_CODE_OAUTH_TOKEN="sk-ant-..." \
  -p 3000:3000 \
  openhands-claude-cli
```

## Kubernetes Deployment

### Using Helm

1. **Create a secret for the OAuth token:**
   ```bash
   kubectl create secret generic claude-oauth-token \
     --from-literal=CLAUDE_CODE_OAUTH_TOKEN="sk-ant-..."
   ```

2. **Install the Helm chart:**
   ```bash
   helm install openhands ./helm/openhands-claude \
     --set claude.existingSecret=claude-oauth-token
   ```

### Values Configuration

```yaml
# values.yaml
claude:
  enabled: true
  model: "claude-sonnet-4-20250514"
  timeout: 300
  existingSecret: "claude-oauth-token"

ingress:
  enabled: true
  hosts:
    - host: openhands.example.com
      paths:
        - path: /
          pathType: Prefix
```

## Limitations

1. **No True Streaming**: The Claude CLI doesn't support streaming, so responses are returned as a single chunk.

2. **Token Estimation**: Token counts are estimated based on character length (approximately 4 characters per token).

3. **Tool Calling**: Tool calls are parsed from the text response using pattern matching. Complex tool use scenarios may require adjustment.

## Troubleshooting

### CLI Not Found

If you see "Claude Code CLI not found", ensure:
- The CLI is installed globally: `npm install -g @anthropic-ai/claude-code`
- The CLI path is correct (check with `which claude`)

### Token Expired

If authentication fails:
1. Run `claude setup-token` to get a new token
2. Update the `CLAUDE_CODE_OAUTH_TOKEN` environment variable
3. Restart the container/pod

### Timeout Errors

If you're getting timeout errors:
- Increase `claude_cli_timeout` in your config
- Check network connectivity to Anthropic's servers

## Development

### Running Tests

```bash
pytest tests/unit/llm/test_claude_cli_transport.py -v
```

### Adding New Features

The transport layer is in `openhands/llm/transports/claude_code_cli.py`. Key classes:

- `ClaudeCodeCLIConfig`: Configuration dataclass
- `ClaudeCodeCLITransport`: Synchronous transport
- `AsyncClaudeCodeCLITransport`: Async transport (uses asyncio)

## Security Considerations

1. **Token Storage**: Never commit OAuth tokens to version control. Use secrets management (Kubernetes secrets, environment variables, etc.).

2. **Network Security**: The CLI makes HTTPS connections to Anthropic. Ensure your network allows outbound HTTPS traffic.

3. **Container Security**: Run containers with minimal privileges. The provided Dockerfile creates a non-root user for running the application.
