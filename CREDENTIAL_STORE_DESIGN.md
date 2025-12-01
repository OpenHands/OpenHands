# Credential Store Integration Design

## Overview

This document outlines the design for implementing credential store integration that allows OpenHands to automatically authenticate with private resources at runtime.

## Problem Statement

Currently, OpenHands can store custom secrets as environment variables, but there's no mechanism to automatically use these credentials when accessing private resources (private npm registries, Docker registries, databases, APIs, etc.). Users must manually configure authentication, which is error-prone and doesn't scale well.

## Goals

1. **Automatic Authentication**: Detect authentication failures (401/403) and automatically retry with stored credentials
2. **Resource-based Credential Mapping**: Map credentials to resources (URLs, domains, service types)
3. **Multiple Auth Methods**: Support API keys, tokens, basic auth, bearer tokens
4. **Runtime Integration**: Seamlessly integrate with existing runtime HTTP clients

## Architecture

### Components

1. **Credential Store Extension**
   - Extend `Secrets` model to include credential mappings
   - Support resource patterns (URL, domain, service type)

2. **Credential Resolver**
   - Match resource URLs/domains to stored credentials
   - Determine authentication method based on credential type

3. **Authentication Middleware**
   - Intercept HTTP requests
   - Detect 401/403 responses
   - Inject credentials and retry

4. **API Endpoints**
   - CRUD operations for credential mappings
   - List available credentials for a resource

### Data Model

```python
class CredentialMapping(BaseModel):
    """Maps a resource to authentication credentials."""
    resource_pattern: str  # URL pattern or domain (e.g., "npm.example.com", "https://api.example.com/*")
    credential_name: str  # Reference to a custom secret
    auth_method: Literal["api_key", "bearer_token", "basic_auth", "header"]  # How to use the credential
    auth_header: str | None = None  # Custom header name (for "header" method)
    resource_type: str | None = None  # Optional: "npm", "docker", "api", etc.
    description: str | None = None
```

Extension to `Secrets` model:
```python
class Secrets(BaseModel):
    provider_tokens: PROVIDER_TOKEN_TYPE = Field(default_factory=lambda: MappingProxyType({}))
    custom_secrets: CUSTOM_SECRETS_TYPE = Field(default_factory=lambda: MappingProxyType({}))
    credential_mappings: MappingProxyType[str, CredentialMapping] = Field(default_factory=lambda: MappingProxyType({}))  # New field
```

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Extend Secrets Model**
   - Add `CredentialMapping` model
   - Add `credential_mappings` field to `Secrets`
   - Update serialization/deserialization

2. **Create Credential Resolver**
   - `CredentialResolver` class that matches URLs to credentials
   - Pattern matching (exact URL, domain, wildcard patterns)
   - Returns appropriate auth headers/credentials

3. **Update Secrets Store**
   - Ensure `credential_mappings` are persisted
   - Migrate existing stores

### Phase 2: Runtime Integration

1. **HTTP Client Interceptor**
   - Wrap existing HTTP clients (`httpx.AsyncClient`)
   - Detect 401/403 responses
   - Resolve credentials and retry

2. **Runtime Environment Setup**
   - Initialize credential resolver in runtime
   - Make credentials available to HTTP clients

3. **Agent Tool Integration**
   - Ensure agent's HTTP requests use credential resolver
   - Support npm, docker, curl commands with auto-auth

### Phase 3: API and UI

1. **API Endpoints**
   - `POST /api/credentials/mappings` - Create mapping
   - `GET /api/credentials/mappings` - List mappings
   - `PUT /api/credentials/mappings/{id}` - Update mapping
   - `DELETE /api/credentials/mappings/{id}` - Delete mapping
   - `GET /api/credentials/resolve?url=...` - Test credential resolution

2. **Frontend Integration**
   - UI for managing credential mappings
   - Resource pattern helper/validator

### Phase 4: Testing and Documentation

1. **Unit Tests**
   - Credential resolver pattern matching
   - HTTP interceptor retry logic
   - API endpoints

2. **Integration Tests**
   - End-to-end authentication flow
   - Multiple credential mappings
   - Conflict resolution

3. **Documentation**
   - User guide for setting up credentials
   - Examples for common services (npm, Docker, etc.)

## Example Usage

### Setting up a credential mapping:

```python
# User creates a custom secret
POST /api/secrets
{
  "name": "NPM_REGISTRY_TOKEN",
  "value": "npm_xxxxxxxxxxxxx",
  "description": "Token for private npm registry"
}

# User creates a credential mapping
POST /api/credentials/mappings
{
  "resource_pattern": "npm.example.com",
  "credential_name": "NPM_REGISTRY_TOKEN",
  "auth_method": "bearer_token",
  "resource_type": "npm"
}
```

### Runtime behavior:

1. Agent tries to install package: `npm install @company/private-package`
2. npm requests authentication from `npm.example.com`
3. Credential resolver matches pattern and retrieves `NPM_REGISTRY_TOKEN`
4. HTTP client automatically adds `Authorization: Bearer npm_xxxxxxxxxxxxx`
5. Request succeeds

## Security Considerations

1. **Credential Storage**: Credentials are stored encrypted (existing infrastructure)
2. **Pattern Matching**: Exact matches prioritized over wildcards
3. **Scope Limitation**: Credentials only used for matching resources
4. **Audit Logging**: Log credential usage (without exposing values)

## Future Enhancements

1. **Credential Templates**: Pre-configured patterns for common services
2. **OAuth2 Support**: Token refresh for OAuth2 credentials
3. **Multi-credential Support**: Try multiple credentials if first fails
4. **Credential Rotation**: Automatic credential rotation
5. **Context-aware Matching**: Use different credentials based on task context
