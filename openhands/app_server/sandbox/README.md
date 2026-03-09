# Sandbox Management

Manages sandbox environments for secure agent execution within OpenHands.

## Overview

Since agents can do things that may harm your system, they are typically run inside a sandbox (like a Docker container). This module provides services for creating, managing, and monitoring these sandbox environments.

## Key Components

- **SandboxService**: Abstract service for sandbox lifecycle management
- **DockerSandboxService**: Docker-based sandbox implementation
- **SandboxSpecService**: Manages sandbox specifications and templates
- **SandboxRouter**: FastAPI router for sandbox endpoints

## Features

- Secure containerized execution environments
- Sandbox lifecycle management (create, start, stop, destroy)
- Multiple sandbox backend support (Docker, Remote, Local)
- User-scoped sandbox access control

## Observability (Laminar)

When running agents via the **hosted Web UI** (e.g. OpenHands Cloud at app.all-hands.dev), you can enable [Laminar](https://laminar.sh/) observability so traces are exported from the agent server.

1. Add a **workspace secret** (custom secret) named exactly `LMNR_PROJECT_API_KEY` with your [Laminar project API key](https://docs.lmnr.ai/tracing/quickstart).
2. Start a conversation in the hosted UI. The app server injects this secret into the agent server environment at sandbox start, so the SDK can export traces to Laminar.

This uses outbound export only (no inbound ports). The secret is resolved from the workspace-level secret store and injected at runtime. Laminar observability continues to work for local runs (CLI/SDK) as before; the hosted Web UI now supports it when you configure the secret.
