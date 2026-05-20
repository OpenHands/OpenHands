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

## Memory-Integrity Callback (OWASP ASI06)

`MemoryIntegrityCallbackProcessor` is a built-in processor that scans
conversation events for signals of agent memory poisoning — the OWASP
[ASI06 Memory Poisoning](https://owasp.org/www-project-agent-memory-guard/)
threat. Findings are written to the existing event-callback result store so
operators can alert and triage with the same tooling they use for every other
callback.

### What it detects (BUILTIN backend, default)

- Prompt-injection markers (e.g. "ignore previous instructions", fake
  `<|system|>` tags, "developer mode enabled").
- Sensitive-material markers (PEM private keys, AWS access-key IDs, GitHub /
  Stripe / Slack token prefixes).
- Size anomalies above a configurable byte threshold.
- An append-only per-conversation SHA-256 fingerprint chain for offline
  forensic comparison.

### Policy

| Policy | Behavior |
|--------|----------|
| `AUDIT` (default) | Record findings in the result `detail`. `SUCCESS` status. |
| `WARN` | Same as `AUDIT` plus a `WARNING`-level log line. |
| `BLOCK` | Findings flip the result status to `ERROR` so operators can trip downstream alerts. |

The agent's true memory/context lives in `openhands-sdk`; this processor
observes events *after* they reach the app server, so it is an
audit/detection layer rather than an in-line "block-before-write" guard.

### Optional OWASP backend

Set `backend = "OWASP_AGENT_MEMORY_GUARD"` to delegate scanning to the
upstream [agent-memory-guard](https://pypi.org/project/agent-memory-guard/)
PyPI package. The package is **not** a project dependency; install it
separately (`pip install agent-memory-guard`) before selecting this backend.
If it is missing, the callback emits a clean `ERROR` result with install
guidance instead of crashing the callback loop.

### Registering a callback

```bash
curl -X POST $OPENHANDS_URL/api/v1/event-callbacks \
  -H 'Content-Type: application/json' \
  -d '{
        "processor": {
          "kind": "MemoryIntegrityCallbackProcessor",
          "policy": "WARN",
          "backend": "BUILTIN"
        }
      }'
```

Filter by `conversation_id` or `event_kind` like any other callback. Read
back results via `GET /api/v1/event-callback-results`.
