# Connecting OpenHands to Native IDE LLMs (Keyless Integration)

This guide explains how to connect OpenHands to your IDE's native, built-in LLMs (like Gemini, Claude, or GPT) using the open-source **OpenHands IDE Bridge** proxy.

This allows you to run OpenHands completely keyless and stop paying for secondary LLM API keys.

## Setup Instructions

1. **Install and Start the IDE Bridge:**
   On your host machine, install the bridge package:
   ```bash
   pip install git+https://github.com/Yuggohel2/openhands-ide-bridge.git
   openhands-ide-proxy
   ```

2. **Configure OpenHands:**
   Start the OpenHands container and point its LLM base URL to the local proxy:
   * Set `LLM_BASE_URL` to `http://host.docker.internal:9999/v1`
   * Set `LLM_MODEL` to `openai/native-ide-model`
   * Set `LLM_API_KEY` to `dummy`

For more advanced settings and registering OpenHands as an MCP server inside your IDE, see the [OpenHands IDE Bridge Repository](https://github.com/Yuggohel2/openhands-ide-bridge).
