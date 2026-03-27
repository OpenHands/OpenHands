from typing import Literal

from fastapi import Request
from server.constants import IS_FEATURE_ENV, IS_LOCAL_ENV, IS_STAGING_ENV
from starlette.datastructures import URL

from openhands.app_server.config import get_global_config


def get_web_url(request: Request):
    web_url = get_global_config().web_url
    if not web_url:
        scheme = 'http' if request.url.hostname == 'localhost' else 'https'
        web_url = f'{scheme}://{request.url.netloc}'
    else:
        web_url = web_url.rstrip('/')
    return web_url


def get_cookie_domain() -> str | None:
    config = get_global_config()
    web_url = config.web_url
    # for now just use the full hostname except for staging stacks.
    return (
        URL(web_url).hostname
        if web_url and not (IS_FEATURE_ENV or IS_STAGING_ENV or IS_LOCAL_ENV)
        else None
    )


def get_cookie_samesite() -> Literal['lax', 'strict']:
    # Use 'strict' in production for maximum CSRF protection
    # Use 'lax' for local development and staging environments
    # Note: For invitation links from emails, the frontend handles acceptance via
    # an authenticated POST request (same-origin), which works with 'strict' cookies
    web_url = get_global_config().web_url
    return (
        'strict'
        if web_url and not (IS_FEATURE_ENV or IS_STAGING_ENV or IS_LOCAL_ENV)
        else 'lax'
    )
