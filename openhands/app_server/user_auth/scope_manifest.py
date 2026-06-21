from enum import Enum

from pydantic import BaseModel


class ApiKeyScopeName(str, Enum):
    FULL = 'full'
    SANDBOX = 'sandbox'

class ScopeInfo(BaseModel):
    name: ApiKeyScopeName
    description: str
    is_default: bool = False
    is_visible_to_users: bool = True
    # If None, grants all permissions. If set, grants only specific permissions.
    permissions: set[str] | None = None

SCOPE_MANIFEST: dict[ApiKeyScopeName, ScopeInfo] = {
    ApiKeyScopeName.FULL: ScopeInfo(
        name=ApiKeyScopeName.FULL,
        description='Full access to all OpenHands operations, including organization management, secrets, and settings.',
        is_default=True,
        permissions=None,
    ),
    ApiKeyScopeName.SANDBOX: ScopeInfo(
        name=ApiKeyScopeName.SANDBOX,
        description='Restricted access strictly for automated sandboxes. Can access runtime settings but cannot manage secrets, API keys, or organizations.',
        is_default=False,
        permissions=set(), # No SaaS permissions
    ),
}

def get_scope_info() -> list[ScopeInfo]:
    """Returns all scopes that are visible to users (for UI rendering)."""
    return [info for info in SCOPE_MANIFEST.values() if info.is_visible_to_users]
