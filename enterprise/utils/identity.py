"""Shared identity helpers for resolving user profile fields from Keycloak claims."""


def resolve_display_name(user_info: dict) -> str | None:
    """Resolve the best available display name from Keycloak user_info claims.

    Fallback chain: name → given_name + family_name → None

    Does NOT fall back to preferred_username/username — callers that need
    a guaranteed non-None value should handle that separately. This keeps
    the helper focused on real-name claims so that the /api/user/info route
    can return name=None when no real name is available, while user_store
    callers can append their own username fallback.
    """
    name = user_info.get('name', '')
    if name and name.strip():
        return name.strip()

    given = user_info.get('given_name', '').strip()
    family = user_info.get('family_name', '').strip()
    combined = f'{given} {family}'.strip()
    if combined:
        return combined

    return None
