# OpenHands Utilities

Common utility functions and helpers for OpenHands app server.

## Overview

This module provides utility functions that are used across OpenHands for common operations like date handling, SQL operations, dynamic imports, async utilities, LLM integration, and more.

## Key Components

- **async_utils**: Async/sync interoperability utilities
- **chunk_localizer**: File chunk localization for code edits
- **environment**: Environment detection (Docker, storage providers)
- **git**: Git branch validation utilities
- **http_session**: HTTP session configuration
- **import_utils**: Dynamic module import utilities
- **jsonpatch_compat**: JSON patch compatibility utilities
- **llm**: LLM model management and configuration
- **sdk_settings_compat**: SDK settings compatibility layer
- **search_utils**: Pagination and search utilities
- **shutdown_listener**: Graceful shutdown signal handling
- **sql_utils**: SQL database operation helpers
- **docker_utils**: Docker environment utilities
- **encryption_key**: Encryption key utilities

## Runtime Implementation Substitution

OpenHands provides an extensibility mechanism through the `get_impl` and `import_from` functions in `import_utils.py`. This mechanism allows applications built on OpenHands to customize behavior by providing their own implementations of OpenHands base classes.

### How It Works

1. Base classes define interfaces through abstract methods and properties
2. Default implementations are provided by OpenHands
3. Applications can provide custom implementations by:
   - Creating a class that inherits from the base class
   - Implementing all required methods
   - Configuring OpenHands to use the custom implementation via configuration

### Example

```python
# In OpenHands base code:
class ConversationManager:
    @abstractmethod
    async def attach_to_conversation(self, sid: str) -> Conversation:
        """Attach to an existing conversation."""

# Default implementation in OpenHands:
class StandaloneConversationManager(ConversationManager):
    async def attach_to_conversation(self, sid: str) -> Conversation:
        # Single-server implementation
        ...

# In your application:
class ClusteredConversationManager(ConversationManager):
    async def attach_to_conversation(self, sid: str) -> Conversation:
        # Custom distributed implementation
        ...

# In configuration:
server_config.conversation_manager_class = 'myapp.ClusteredConversationManager'
```

### Common Extension Points

OpenHands provides several components that can be extended:

1. Server Components:
   - `ConversationManager`: Manages conversation lifecycles
   - `UserAuth`: Handles user authentication
   - `MonitoringListener`: Provides monitoring capabilities

2. Storage:
   - `ConversationStore`: Stores conversation data
   - `SettingsStore`: Manages user settings
   - `SecretsStore`: Handles sensitive data

3. Service Integrations:
   - GitHub service
   - GitLab service
   - Azure DevOps service

### Implementation Details

The mechanism is implemented through two key functions:

1. `import_from(qual_name: str)`: Imports any Python value from its fully qualified name
   ```python
   UserAuth = import_from('openhands.app_server.user_auth.UserAuth')
   ```

2. `get_impl(cls: type[T], impl_name: str | None) -> type[T]`: Imports and validates a class implementation
   ```python
   ConversationManagerImpl = get_impl(
       ConversationManager,
       server_config.conversation_manager_class
   )
   ```

The `get_impl` function ensures type safety by validating that the imported class is either the same as or a subclass of the specified base class. It also caches results to avoid repeated imports.

## Migration Guide

If upgrading from a version where these utilities were in `openhands.utils`, update your imports:

```python
# Before
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.llm import get_supported_llm_models
from openhands.utils.import_utils import get_impl
from openhands.utils.environment import is_running_in_docker

# After
from openhands.app_server.utils.async_utils import call_sync_from_async
from openhands.app_server.utils.llm import get_supported_llm_models
from openhands.app_server.utils.import_utils import get_impl
from openhands.app_server.utils.environment import is_running_in_docker
```

All utilities previously in `openhands.utils.*` are now available at `openhands.app_server.utils.*`.
