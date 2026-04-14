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

## Built-in processors

- **SetTitleCallbackProcessor** — polls the agent server for a generated
  conversation title and persists it onto the conversation info.
- **FinishCriticCallbackProcessor** — runs a critic
  (`openhands.critic.finish_critic.ExecutionStatusCritic`) when a
  conversation reaches a terminal execution status (`finished`, `error`,
  `stuck`). The resulting `CriticResult` is persisted on the
  `AppConversationInfo` (`critic_result` field, backed by the
  `critic_score`, `critic_message` and `critic_evaluated_at` columns on
  `conversation_metadata`) and the callback marks itself `COMPLETED` so
  each conversation is only scored once. Existing conversation APIs
  return the score as part of the conversation info — no breaking
  contract change is required.

Both processors are installed automatically when a new conversation is
started through the V1 live-status app-conversation service, so every
new conversation produces a live critic score as soon as it completes.
