"""Chunked cookie helpers for the ``keycloak_auth`` session cookie.

Browsers cap a single cookie at ~4096 bytes (name + value + attributes).
The ``keycloak_auth`` cookie wraps a signed JWS containing the Keycloak
access and refresh tokens, and that can exceed the cap for users with
large claim sets (long emails, many realm roles, several allowed-origins
entries). Chrome silently drops an oversized cookie, which shows up as an
endless login loop: the OAuth callback "succeeds" but the cookie never
reaches the next request.

These helpers split an oversized value across numbered sibling cookies
(``keycloak_auth``, ``keycloak_auth_1``, ``keycloak_auth_2`` ...) and
reassemble it on read. A value that fits in one chunk is written as a
single bare cookie, byte-for-byte identical to the previous behaviour, so
sessions that exist today are unaffected.
"""

from __future__ import annotations

from fastapi import Request, Response

# Keep each chunk well under the 4096-byte cap after the cookie name and
# attributes (Domain, Path, Secure, HttpOnly, SameSite) are added.
CHUNK_SIZE = 3000
# Upper bound on how many chunks we will ever write or clear. 8 * 3000 =
# 24KB, far above any realistic token, and well within per-domain cookie
# limits. Reads stop at the first missing chunk, so this only bounds the
# stale-chunk cleanup on write/delete.
MAX_CHUNKS = 8


def _chunk_key(key: str, index: int) -> str:
    """Chunk 0 keeps the bare key so single-cookie writes are unchanged."""
    return key if index == 0 else f'{key}_{index}'


def read_chunked_cookie(request: Request, key: str) -> str | None:
    """Reassemble a possibly-chunked cookie value, or ``None`` if absent.

    Concatenates ``key``, ``key_1``, ``key_2`` ... in order, stopping at
    the first missing index. A plain single cookie reads back unchanged.
    """
    first = request.cookies.get(key)
    if first is None:
        return None
    parts = [first]
    for i in range(1, MAX_CHUNKS):
        part = request.cookies.get(_chunk_key(key, i))
        if part is None:
            break
        parts.append(part)
    return ''.join(parts)


def set_chunked_cookie(
    response: Response,
    key: str,
    value: str,
    *,
    domain: str | None = None,
    secure: bool = True,
    httponly: bool = True,
    samesite: str = 'lax',
) -> None:
    """Set ``key`` to ``value``, splitting across sibling cookies if needed.

    Always clears trailing chunk indices that this write does not use, so a
    value that shrank from N chunks to fewer does not leave stale chunks
    that ``read_chunked_cookie`` would wrongly append.
    """
    chunks = [value[i : i + CHUNK_SIZE] for i in range(0, len(value), CHUNK_SIZE)] or [
        ''
    ]

    for i, chunk in enumerate(chunks):
        kwargs: dict = {
            'httponly': httponly,
            'secure': secure,
            'samesite': samesite,
        }
        if domain:
            kwargs['domain'] = domain
        response.set_cookie(key=_chunk_key(key, i), value=chunk, **kwargs)

    # Expire any chunks left over from a previously larger value.
    for i in range(len(chunks), MAX_CHUNKS):
        _delete_one(response, _chunk_key(key, i), domain=domain, samesite=samesite)


def delete_chunked_cookie(
    response: Response,
    key: str,
    *,
    domain: str | None = None,
    samesite: str = 'lax',
) -> None:
    """Delete ``key`` and every sibling chunk."""
    for i in range(MAX_CHUNKS):
        _delete_one(response, _chunk_key(key, i), domain=domain, samesite=samesite)


def _delete_one(
    response: Response,
    key: str,
    *,
    domain: str | None = None,
    samesite: str = 'lax',
) -> None:
    kwargs: dict = {'samesite': samesite}
    if domain:
        kwargs['domain'] = domain
    response.delete_cookie(key=key, **kwargs)
