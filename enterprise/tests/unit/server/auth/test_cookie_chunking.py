"""Tests for the chunked ``keycloak_auth`` cookie helpers.

These exercise the write/read/delete round trip against a real Starlette
``Response`` (so the actual ``Set-Cookie`` headers are produced) and a
browser-like cookie jar reconstructed from those headers.
"""

from http.cookies import SimpleCookie
from types import SimpleNamespace

import pytest
from server.auth.cookie_chunking import (
    CHUNK_SIZE,
    delete_chunked_cookie,
    read_chunked_cookie,
    set_chunked_cookie,
)
from starlette.responses import Response


def _cookie_jar(response: Response) -> dict[str, str]:
    """Reduce a response's Set-Cookie headers to what a browser would keep."""
    jar: dict[str, str] = {}
    for header in response.headers.getlist('set-cookie'):
        parsed: SimpleCookie = SimpleCookie()
        parsed.load(header)
        for name, morsel in parsed.items():
            # A deletion is emitted as an empty value with Max-Age=0.
            if morsel.value == '' and str(morsel['max-age']) == '0':
                jar.pop(name, None)
            else:
                jar[name] = morsel.value
    return jar


def _request_with(jar: dict[str, str]) -> SimpleNamespace:
    # read_chunked_cookie only touches request.cookies.get(...)
    return SimpleNamespace(cookies=dict(jar))


def _roundtrip(value: str) -> tuple[str | None, dict[str, str]]:
    resp = Response()
    set_chunked_cookie(resp, 'keycloak_auth', value, domain='example.com')
    jar = _cookie_jar(resp)
    return read_chunked_cookie(_request_with(jar), 'keycloak_auth'), jar


def test_small_value_uses_single_cookie_and_roundtrips():
    value = 'x' * 100
    got, jar = _roundtrip(value)
    assert got == value
    assert set(jar) == {'keycloak_auth'}


def test_large_value_splits_into_chunks_each_under_cap():
    # ~4125 bytes is the testadmin2-sized token that exceeds Chrome's 4096 cap.
    value = 'y' * 4125
    got, jar = _roundtrip(value)
    assert got == value
    assert 'keycloak_auth' in jar and 'keycloak_auth_1' in jar
    assert all(len(v) <= CHUNK_SIZE for v in jar.values())


def test_absent_cookie_reads_none():
    assert read_chunked_cookie(_request_with({}), 'keycloak_auth') is None


def test_backward_compatible_with_legacy_single_cookie():
    # A session created before chunking stored one bare cookie.
    got = read_chunked_cookie(
        _request_with({'keycloak_auth': 'legacy-token'}), 'keycloak_auth'
    )
    assert got == 'legacy-token'


def test_shrinking_value_clears_stale_chunks():
    big = Response()
    set_chunked_cookie(big, 'keycloak_auth', 'y' * 4125, domain='example.com')
    assert 'keycloak_auth_1' in _cookie_jar(big)

    # A later, smaller token must not leave keycloak_auth_1 behind, or the
    # reader would append a stale chunk and corrupt the token.
    small = Response()
    set_chunked_cookie(small, 'keycloak_auth', 'x' * 100, domain='example.com')
    assert set(_cookie_jar(small)) == {'keycloak_auth'}


def test_delete_expires_base_and_siblings():
    resp = Response()
    delete_chunked_cookie(resp, 'keycloak_auth', domain='example.com')
    headers = '\n'.join(resp.headers.getlist('set-cookie'))
    assert 'keycloak_auth=' in headers
    assert 'keycloak_auth_1=' in headers


@pytest.mark.parametrize('size', [CHUNK_SIZE, CHUNK_SIZE + 1, CHUNK_SIZE * 2 + 7])
def test_boundary_sizes_roundtrip(size):
    value = 'z' * size
    got, _ = _roundtrip(value)
    assert got == value
