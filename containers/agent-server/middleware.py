"""
Custom CORS Middleware for Agent Server
This middleware supports PERMITTED_CORS_ORIGINS environment variable
"""
import os
from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp


class CustomCORSMiddleware(CORSMiddleware):
    """Custom CORS middleware that supports PERMITTED_CORS_ORIGINS environment variable."""

    def __init__(self, app: ASGIApp) -> None:
        allow_origins_str = os.getenv('PERMITTED_CORS_ORIGINS', '')
        if allow_origins_str:
            allow_origins = tuple(
                origin.strip() for origin in allow_origins_str.split(',')
            )
        else:
            allow_origins = ()
        
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

    def is_allowed_origin(self, origin: str) -> bool:
        # First, check if we have explicit allow_origins configured
        if self.allow_origins:
            # Check if the origin is in the allowed list
            if origin in self.allow_origins:
                return True
        
        # If no explicit allow_origins, allow localhost/127.0.0.1
        if origin and not self.allow_origins and not self.allow_origin_regex:
            parsed = urlparse(origin)
            hostname = parsed.hostname or ''

            # Allow any localhost/127.0.0.1 origin regardless of port
            if hostname in ['localhost', '127.0.0.1']:
                return True

        # For missing origin or other origins, use the parent class's logic
        result: bool = super().is_allowed_origin(origin)
        return result

