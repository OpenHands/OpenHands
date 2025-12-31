# User Management

Handles user authentication, authorization, and profile management for the OpenHands app server.

## Overview

This module provides user management capabilities, including authentication, user profile access, and service resolution for user-scoped operations.

## Key Components

- **UserContext**: Abstract context for user operations
- **AuthUserContext**: Compatibility layer for user auth.
- **UserRouter**: FastAPI router for user-related endpoints
- **UserContextInjector**: Factory for getting user context with FastAPI dependency injection

## Features

- User authentication and session management
- Current user profile retrieval
- User-scoped service resolution
- JWT-based authentication integration
