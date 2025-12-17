# Sharing Package

This package contains functionality for sharing conversations.

## Components

- **shared.py**: Data models for shared conversations
- **shared_conversation_info_service.py**: Service interface for accessing shared conversation info
- **sql_shared_conversation_info_service.py**: SQL implementation of the shared conversation info service
- **shared_event_service.py**: Service interface for accessing shared events
- **shared_event_service_impl.py**: Implementation of the shared event service
- **shared_conversation_router.py**: REST API endpoints for shared conversations
- **shared_event_router.py**: REST API endpoints for shared events

## Features

- Read-only access to shared conversations
- Event access for shared conversations
- Search and filtering capabilities
- Pagination support
