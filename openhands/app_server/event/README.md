# Event Management

Handles event storage, retrieval, and streaming for the OpenHands app server.

## Overview

This module provides services for managing events within conversations, including event persistence, querying, and real-time streaming capabilities.

## Key Components

- **EventService**: Abstract service for event CRUD operations
- **FilesystemEventService**: File-based event storage implementation
- **EventRouter**: FastAPI router for event-related endpoints

## Features

- Event storage and retrieval by conversation ID
- Event filtering by kind, timestamp, and other criteria
- Sorting support and pagination for large event sets
- Real-time event streaming capabilities
- Multiple storage backend support (filesystem, database)
