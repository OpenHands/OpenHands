# Sharing Package

This package contains functionality for sharing conversations publicly.

## Components

- **public_conversation_models.py**: Data models for public conversations
- **public_conversation_info_service.py**: Service interface for accessing public conversation info
- **sql_public_conversation_info_service.py**: SQL implementation of the public conversation info service
- **public_event_service.py**: Service interface for accessing public events
- **public_event_service_impl.py**: Implementation of the public event service
- **public_conversation_router.py**: REST API endpoints for public conversations
- **public_event_router.py**: REST API endpoints for public events

## Features

- Read-only access to public conversations
- Event access for public conversations
- Search and filtering capabilities
- Pagination support