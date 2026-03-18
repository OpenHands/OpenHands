# System Architecture Overview

OpenHands supports multiple deployment configurations. This document describes the core components and how they interact.

## Local/Docker Deployment

The simplest deployment runs everything locally or in Docker containers:

```mermaid
flowchart TB
    subgraph Server["OpenHands Server"]
        API["REST API<br/>(FastAPI)"]
        ConvMgr["Conversation<br/>Manager"]
        Runtime["Runtime<br/>Manager"]
    end

    subgraph Sandbox["Sandbox (Docker Container)"]
        AES["Action Execution<br/>Server"]
        Browser["Browser<br/>Environment"]
        FS["File System"]
    end

    User["User"] -->|"HTTP/WebSocket"| API
    API --> ConvMgr
    ConvMgr --> Runtime
    Runtime -->|"Provision"| Sandbox

    Server -->|"Execute actions"| AES
    AES --> Browser
    AES --> FS
```

### Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Server** | REST API, conversation management, runtime orchestration | `openhands/server/` |
| **Runtime** | Abstract interface for sandbox execution | `openhands/runtime/` |
| **Action Execution Server** | Execute bash, file ops, browser actions | Inside sandbox |
| **EventStream** | Central event bus for all communication | `openhands/events/` |

## Scalable Deployment

For production deployments, OpenHands can be configured with a separate Runtime API service:

```mermaid
flowchart TB
    subgraph AppServer["App Server"]
        API["REST API"]
        ConvMgr["Conversation<br/>Manager"]
    end

    subgraph RuntimeAPI["Runtime API (Optional)"]
        RuntimeMgr["Runtime<br/>Manager"]
        WarmPool["Warm Pool"]
    end

    subgraph Sandbox["Sandbox"]
        AS["Agent Server"]
        AES["Action Execution<br/>Server"]
    end

    User["User"] -->|"HTTP"| API
    API --> ConvMgr
    ConvMgr -->|"Provision"| RuntimeMgr
    RuntimeMgr --> WarmPool
    RuntimeMgr --> Sandbox

    User -.->|"WebSocket"| AS
    AS -->|"HTTP"| AES
```

This configuration enables:
- **Warm pool**: Pre-provisioned runtimes for faster startup
- **Direct WebSocket**: Users connect directly to their sandbox, bypassing the App Server
- **Horizontal scaling**: App Server and Runtime API can scale independently

### Runtime Options

OpenHands supports multiple runtime implementations:

| Runtime | Use Case |
|---------|----------|
| **DockerRuntime** | Local development, single-machine deployments |
| **RemoteRuntime** | Connect to externally managed sandboxes |
| **ModalRuntime** | Serverless execution via Modal |

See the [Runtime documentation](https://docs.openhands.dev/usage/architecture/runtime) for details.
