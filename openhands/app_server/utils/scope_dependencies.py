from fastapi import Depends, HTTPException, Request, status

from openhands.app_server.user_auth import get_user_id
from openhands.app_server.user_auth.scope_manifest import SCOPE_MANIFEST


def require_scope(allowed_scopes: list[str]):
    """
    Factory function that creates a dependency to require a specific API key scope.
    This is useful for protecting OSS routes that don't use SaaS permission checks.
    """
    async def scope_checker(
        request: Request,
        user_id: str | None = Depends(get_user_id),
    ) -> str | None:

        user_auth = getattr(request.state, 'user_auth', None)
        api_key_scopes = getattr(user_auth, 'api_key_scopes', None)

        # If it's a cookie auth or old token without scopes, grant full access by default.
        if api_key_scopes is None:
            return user_id

        # Check if the API key has at least one of the allowed scopes
        has_allowed_scope = False
        for scope_name in api_key_scopes:
            if scope_name in allowed_scopes:
                has_allowed_scope = True
                break

            # If the scope allows all (permissions=None) and we are protecting an endpoint,
            # we consider 'full' to inherently satisfy any scope requirement.
            scope_info = SCOPE_MANIFEST.get(scope_name)
            if scope_info and scope_info.permissions is None:
                has_allowed_scope = True
                break

        if not has_allowed_scope:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f'API key scope restricts access to this endpoint. Requires one of: {allowed_scopes}',
            )

        return user_id

    return scope_checker
