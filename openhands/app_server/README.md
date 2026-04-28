# OpenHands App Server

FastAPI-based application server that provides REST API endpoints for OpenHands V1 integration.

## Overview

As of 2025-09-29, much of the code in the OpenHands repository can be regarded as legacy, having been superseded by the code in AgentSDK. This package provides endpoints to interface with the new agent SDK and bridge the gap with the existing OpenHands project.

## Architecture

The app server is organized into several key modules:

- **app_conversation/**: Manages sandboxed conversations and their lifecycle
- **app_lifespan/**: Application startup and shutdown lifecycle management
- **config_api/**: Configuration API endpoints
- **event/**: Handles event storage, retrieval, and streaming
- **event_callback/**: Manages webhooks and event callbacks
- **git/**: Git integration endpoints
- **pending_messages/**: Server-side message queuing
- **sandbox/**: Manages sandbox environments for agent execution
- **secrets/**: Secrets management endpoints
- **services/**: Core services like JWT authentication
- **settings/**: User and application settings endpoints
- **status/**: Server status and system stats
- **user/**: User management and authentication
- **utils/**: Utility functions for common operations
- **web_client/**: Web client configuration and routing
