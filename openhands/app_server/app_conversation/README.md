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
