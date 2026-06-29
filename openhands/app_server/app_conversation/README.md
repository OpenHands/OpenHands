# Conversation Management

Manages app conversations and their lifecycle within the OpenHands app server.

## Overview

This module provides services and models for managing conversations that run within sandboxed environments. It handles conversation creation, retrieval, status tracking, and lifecycle management.

## Key Components

- **AppConversationService**: Abstract service for conversation CRUD operations
- **LiveStatusAppConversationService**: Real-time conversation status tracking
- **AppConversationRouter**: FastAPI router for conversation endpoints

## Features

- Conversation search and filtering by title, dates, and status
- Real-time conversation status updates
- Pagination support for large conversation lists
- Integration with sandbox environments
- Trajectory exports include raw event JSON plus an
  `evidence_gate_receipts.json` projection that links each action event to its
  security risk, pre-tool hook decisions, observations, and post-tool hooks for
  reviewer-facing audit evidence.
  The receipt decision is a derived export projection from security risk and
  hook block state, not a replacement for the runtime confirmation policy.
