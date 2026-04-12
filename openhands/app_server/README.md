# OpenHands App Server

FastAPI-based application server that provides REST API endpoints for OpenHands V1 integration.

## Overview

As of 2025-09-29, much of the code in the OpenHands repository can be regarded as legacy, having been superseded by the code in AgentSDK. This package provides endpoints to interface with the new agent SDK and bridge the gap with the existing OpenHands project.

## Architecture

The app server is organized into several key modules:

- **conversation/**: Manages sandboxed conversations and their lifecycle
- **event/**: Handles event storage, retrieval, and streaming
- **event_callback/**: Manages webhooks and event callbacks
- **sandbox/**: Manages sandbox environments for agent execution
- **user/**: User management and authentication
- **services/**: Core services like JWT authentication
- **utils/**: Utility functions for common operations

## Docker TOML Configuration

When running the app server in Docker, configuration can be provided through a
TOML file instead of only environment variables.

Resolution order:

1. Path from `OH_CONFIG_FILE` (or `OPENHANDS_CONFIG_FILE`)
2. `docker.toml` in the current working directory
3. `config.toml` in the current working directory
4. `/app/docker.toml`
5. `/app/config.toml`

The app server supports both sections:

- `[app_server]` and `[app_server.sandbox]` (V1 format)
- `[sandbox]` (legacy-compatible format)

Environment variables still take precedence over TOML values.

Example:

```toml
[app_server]
web_url = "https://openhands.example.com"
permitted_cors_origins = ["https://frontend.example.com"]

[app_server.sandbox]
use_host_network = false
volumes = "/host/workspace:/workspace:rw"

# Legacy compatibility for exposing additional sandbox ports
runtime_extra_build_args = ["-p 80:80", "--publish=8080:8080"]
```
