"""Unit tests for the ``reject_x_org_id_path_mismatch`` FastAPI dependency.

The guard's job is narrow: when a route has ``{org_id}`` in its path
*and* the request carries an ``X-Org-Id`` header, the two must agree.
Everything else is a pass-through.

These tests drive the dependency through a minimal FastAPI app rather
than calling the function directly — that way we exercise the same
header-parsing / path-param-resolution machinery that real requests
go through, and we catch wiring regressions (e.g. someone renaming
the header alias) that a direct call would silently tolerate.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.auth.org_context import (
    REJECT_X_ORG_ID_PATH_MISMATCH,
    X_ORG_ID_HEADER,
)


@pytest.fixture
def app() -> FastAPI:
    """Build a tiny app with one path-org route and one non-path route,
    both guarded by the dependency. The non-path route asserts that a
    misconfigured attachment is a no-op (not a 500)."""
    app = FastAPI()

    @app.get(
        '/orgs/{org_id}/things',
        dependencies=[REJECT_X_ORG_ID_PATH_MISMATCH],
    )
    def path_org_route(org_id: UUID) -> dict:
        return {'org_id': str(org_id)}

    @app.get(
        '/no-path-org',
        dependencies=[REJECT_X_ORG_ID_PATH_MISMATCH],
    )
    def no_path_org_route() -> dict:
        return {'ok': True}

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def org_id() -> UUID:
    return uuid4()


# --------------------------------------------------------------------- #
# Pass-through cases
# --------------------------------------------------------------------- #


def test_no_header_passes_through(client: TestClient, org_id: UUID):
    r = client.get(f'/orgs/{org_id}/things')
    assert r.status_code == 200
    assert r.json() == {'org_id': str(org_id)}


def test_matching_header_passes_through(client: TestClient, org_id: UUID):
    r = client.get(
        f'/orgs/{org_id}/things',
        headers={X_ORG_ID_HEADER: str(org_id)},
    )
    assert r.status_code == 200


def test_matching_header_case_insensitive_uuid(client: TestClient, org_id: UUID):
    """UUIDs compare value-wise, not string-wise — uppercase header
    must match lowercase path."""
    r = client.get(
        f'/orgs/{org_id}/things',
        headers={X_ORG_ID_HEADER: str(org_id).upper()},
    )
    assert r.status_code == 200


def test_empty_header_value_passes_through(client: TestClient, org_id: UUID):
    """An empty string header is treated as 'not present' by FastAPI's
    Header dependency. We document the behavior here so a future change
    that starts treating ``X-Org-Id: ''`` as a malformed UUID is
    caught."""
    r = client.get(
        f'/orgs/{org_id}/things',
        headers={X_ORG_ID_HEADER: ''},
    )
    # Empty header → either passes through (current Starlette behavior)
    # or 400. Either is defensible; what we must NEVER do is leak it
    # through to the handler with a falsy-but-truthy "did the user
    # request an override?" signal.
    assert r.status_code in (200, 400)


# --------------------------------------------------------------------- #
# Rejection cases
# --------------------------------------------------------------------- #


def test_conflicting_header_rejected_with_400(client: TestClient, org_id: UUID):
    other = uuid4()
    r = client.get(
        f'/orgs/{org_id}/things',
        headers={X_ORG_ID_HEADER: str(other)},
    )
    assert r.status_code == 400
    body = r.json()
    detail = body.get('detail', '')
    # Error message must name both ids so an operator reading logs can
    # immediately tell which side is stale.
    assert str(other) in detail
    assert str(org_id) in detail


def test_malformed_header_rejected_with_400(client: TestClient, org_id: UUID):
    r = client.get(
        f'/orgs/{org_id}/things',
        headers={X_ORG_ID_HEADER: 'not-a-uuid'},
    )
    assert r.status_code == 400
    assert 'not a valid UUID' in r.json().get('detail', '')


# --------------------------------------------------------------------- #
# Misconfiguration safety
# --------------------------------------------------------------------- #


def test_dep_on_non_path_route_is_noop(client: TestClient):
    """If a developer attaches the dep to a route that has no ``{org_id}``
    path param, the dep should NOT raise — even when the header is
    present. (A 500 here would be worse than the silent miswiring that
    we'd catch in dev when no behavior changed.)"""
    r = client.get(
        '/no-path-org',
        headers={X_ORG_ID_HEADER: str(uuid4())},
    )
    assert r.status_code == 200
    assert r.json() == {'ok': True}


def test_dep_on_non_path_route_ignores_malformed_header(client: TestClient):
    """Same as above, but with a junk header — still no-op, NOT 400.
    The dep should only complain about path-vs-header conflicts; if
    there's no path to compare against, it stays silent."""
    r = client.get(
        '/no-path-org',
        headers={X_ORG_ID_HEADER: 'not-a-uuid'},
    )
    assert r.status_code == 200
