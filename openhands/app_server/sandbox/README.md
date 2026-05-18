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
- Multiple sandbox backend support (Docker, Remote, Local, boxd)
- User-scoped sandbox access control

## Backends

| `RUNTIME=` | Class | Isolation | Use case |
|---|---|---|---|
| `docker` (default) | `DockerSandboxService` | Docker container | Local dev, single machine |
| `remote` | `RemoteSandboxService` | All-Hands managed runtime | Managed cloud |
| `local` / `process` | `ProcessSandboxService` | Local process | Minimal isolation, dev only |
| `boxd` | `BoxdSandboxService` | KVM microVM | Self-hostable cloud with per-VM IPv4 + sub-ms warm suspend/resume |

See `boxd-runtime.md` for the boxd setup guide.
