# Credential Store Integration - Implementation Summary

## Issue #11844: Complete Implementation

This document summarizes the complete implementation of credential store integration for automatic authentication with private resources at runtime.

## Implementation Overview

### Components Implemented

1. **Data Models**
   - `CredentialMapping` - Maps resource patterns to credentials
   - Extended `Secrets` model to include `credential_mappings` field
   - Full serialization/deserialization support with backward compatibility

2. **Credential Resolver**
   - `CredentialResolver` class that matches URLs to stored credentials
   - Supports exact, domain, and wildcard pattern matching
   - Builds appropriate authentication headers based on auth method

3. **API Endpoints**
   - `GET /api/credentials/mappings` - List all credential mappings
   - `POST /api/credentials/mappings` - Create a new credential mapping
   - `PUT /api/credentials/mappings/{id}` - Update an existing mapping
   - `DELETE /api/credentials/mappings/{id}` - Delete a mapping
   - `GET /api/credentials/resolve?url=...` - Test credential resolution

4. **HTTP Client Integration**
   - `send_request_with_credential_retry` utility for automatic credential injection
   - Detects 401/403 errors and automatically retries with credentials

5. **Comprehensive Tests**
   - Unit tests for CredentialMapping model
   - Unit tests for CredentialResolver
   - Unit tests for API endpoints
   - Integration tests for Secrets model with credential mappings

## Files Created/Modified

### New Files
- `openhands/storage/data_models/credential_mapping.py` - CredentialMapping model
- `openhands/storage/credentials/resolver.py` - CredentialResolver implementation
- `openhands/storage/credentials/__init__.py` - Module exports
- `openhands/utils/credential_injector.py` - HTTP request wrapper with credential injection
- `tests/unit/storage/data_models/test_credential_mapping.py` - CredentialMapping tests
- `tests/unit/storage/credentials/test_credential_resolver.py` - CredentialResolver tests
- `tests/unit/server/routes/test_credential_mappings_api.py` - API endpoint tests
- `tests/unit/storage/data_models/test_secrets_credential_mappings.py` - Secrets integration tests

### Modified Files
- `openhands/storage/data_models/secrets.py` - Added credential_mappings field
- `openhands/server/settings.py` - Added CredentialMappingModel and GETCredentialMappings
- `openhands/server/routes/secrets.py` - Added credential mapping API endpoints
- `openhands/runtime/utils/request.py` - Added credential injection import

## Usage Examples

### 1. Creating a Credential Mapping

**Via API:**
```bash
# First, create a custom secret
POST /api/secrets
{
  "name": "NPM_REGISTRY_TOKEN",
  "value": "npm_xxxxxxxxxxxxx",
  "description": "Token for private npm registry"
}

# Then, create a credential mapping
POST /api/credentials/mappings
{
  "resource_pattern": "npm.example.com",
  "credential_name": "NPM_REGISTRY_TOKEN",
  "auth_method": "bearer_token",
  "resource_type": "npm",
  "description": "NPM registry authentication"
}
```

**In Python:**
```python
from openhands.storage.data_models.credential_mapping import CredentialMapping
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.credentials.resolver import CredentialResolver

# Create mapping
mapping = CredentialMapping(
    resource_pattern='npm.example.com',
    credential_name='NPM_REGISTRY_TOKEN',
    auth_method='bearer_token',
    resource_type='npm'
)

# Resolve credentials for a URL
resolver = CredentialResolver(secrets)
result = resolver.resolve_credential('https://npm.example.com/package')
if result:
    credential_value, auth_headers = result
    # auth_headers contains {'Authorization': 'Bearer npm_xxxxxxxxxxxxx'}
```

### 2. Supported Authentication Methods

- **bearer_token**: `Authorization: Bearer <token>`
- **api_key**: Custom header (default: `X-API-Key: <key>`) or custom header via `auth_header`
- **basic_auth**: `Authorization: Basic <base64(username:password)>`
- **header**: Custom header name specified via `auth_header` field

### 3. Pattern Matching Examples

```python
# Exact match
resource_pattern: "https://api.example.com"
matches: "https://api.example.com", "https://api.example.com/v1/endpoint"

# Domain match
resource_pattern: "api.example.com"
matches: "api.example.com", "https://api.example.com", "http://api.example.com/v1"

# Wildcard pattern
resource_pattern: "https://api.example.com/*"
matches: "https://api.example.com/v1", "https://api.example.com/v2/resource"
```

### 4. Automatic HTTP Request Retry

```python
from openhands.utils.credential_injector import send_request_with_credential_retry
from openhands.storage.credentials.resolver import CredentialResolver

resolver = CredentialResolver(secrets)

# This will automatically retry with credentials if 401/403 occurs
response = await send_request_with_credential_retry(
    resolver=resolver,
    session=httpx_client,
    method='GET',
    url='https://private-api.example.com/resource'
)
```

## Testing

Run the tests with:

```bash
# Test credential mapping model
pytest tests/unit/storage/data_models/test_credential_mapping.py -v

# Test credential resolver
pytest tests/unit/storage/credentials/test_credential_resolver.py -v

# Test API endpoints
pytest tests/unit/server/routes/test_credential_mappings_api.py -v

# Test Secrets integration
pytest tests/unit/storage/data_models/test_secrets_credential_mappings.py -v
```

## Backward Compatibility

- The `credential_mappings` field in `Secrets` defaults to an empty MappingProxyType
- Existing code that doesn't use credential mappings will continue to work
- Deserialization handles missing `credential_mappings` field gracefully

## Security Considerations

1. **Credential Storage**: Credentials are stored encrypted using existing infrastructure
2. **Pattern Matching**: Exact matches are prioritized over wildcards to prevent credential leakage
3. **Scope Limitation**: Credentials are only used for matching resource patterns
4. **API Security**: All endpoints require authentication (via `get_dependencies()`)

## Future Enhancements

Potential future improvements:
1. OAuth2 token refresh support
2. Credential rotation automation
3. Multi-credential fallback (try multiple credentials if first fails)
4. Context-aware credential selection
5. Pre-configured templates for common services (npm, Docker Hub, etc.)

## Documentation

- Design document: `CREDENTIAL_STORE_DESIGN.md`
- This implementation summary: `CREDENTIAL_STORE_IMPLEMENTATION.md`

## Status

✅ **Complete** - All components implemented and tested:
- Data models
- Credential resolver
- API endpoints
- HTTP client integration
- Comprehensive test suite

The implementation is ready for use and can be extended with additional features as needed.
