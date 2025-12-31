# Event Callbacks

Manages webhooks and event callbacks for external system integration.

## Overview

This module provides webhook and callback functionality, allowing external systems to receive notifications when specific events occur within OpenHands conversations.

## Key Components

- **EventCallbackService**: Abstract service for callback CRUD operations
- **SqlEventCallbackService**: SQL-based callback storage implementation
- **EventWebhookRouter**: FastAPI router for webhook endpoints

## Features

- Webhook registration and management
- Event filtering by type and conversation
- Callback result tracking and status monitoring
- Retry logic for failed webhook deliveries
- Secure webhook authentication
