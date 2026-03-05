from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.routes.bitbucket_dc_proxy import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with patch(
        'server.routes.bitbucket_dc_proxy.BITBUCKET_DATA_CENTER_HOST', 'bitbucket.test'
    ):
        yield TestClient(app)


def test_missing_authorization_header(client):
    response = client.get('/bitbucket-dc-proxy/oauth2/userinfo')
    assert response.status_code == 401
    assert response.json() == {'error': 'missing_token'}


def test_non_bearer_scheme(client):
    response = client.get(
        '/bitbucket-dc-proxy/oauth2/userinfo',
        headers={'Authorization': 'Basic xyz'},
    )
    assert response.status_code == 401
    assert response.json() == {'error': 'missing_token'}


def test_whoami_non_200(client):
    whoami_resp = MagicMock()
    whoami_resp.status_code = 403

    with patch('server.routes.bitbucket_dc_proxy.httpx.AsyncClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_resp])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        response = client.get(
            '/bitbucket-dc-proxy/oauth2/userinfo',
            headers={'Authorization': 'Bearer some_token'},
        )

    assert response.status_code == 401
    assert response.json() == {'error': 'not_authenticated'}


def test_whoami_empty_body(client):
    whoami_resp = MagicMock()
    whoami_resp.status_code = 200
    whoami_resp.text = '   '

    with patch('server.routes.bitbucket_dc_proxy.httpx.AsyncClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_resp])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        response = client.get(
            '/bitbucket-dc-proxy/oauth2/userinfo',
            headers={'Authorization': 'Bearer some_token'},
        )

    assert response.status_code == 401
    assert response.json() == {'error': 'not_authenticated'}


def test_user_details_non_200(client):
    whoami_resp = MagicMock()
    whoami_resp.status_code = 200
    whoami_resp.text = 'testuser'

    user_resp = MagicMock()
    user_resp.status_code = 404

    with patch('server.routes.bitbucket_dc_proxy.httpx.AsyncClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_resp, user_resp])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        response = client.get(
            '/bitbucket-dc-proxy/oauth2/userinfo',
            headers={'Authorization': 'Bearer some_token'},
        )

    assert response.status_code == 404
    assert response.json() == {'error': 'bitbucket_error: 404'}


def test_happy_path_full_user_data(client):
    whoami_resp = MagicMock()
    whoami_resp.status_code = 200
    whoami_resp.text = 'jsmith'

    user_resp = MagicMock()
    user_resp.status_code = 200
    user_resp.json.return_value = {
        'id': 42,
        'name': 'jsmith',
        'displayName': 'John Smith',
        'emailAddress': 'john@example.com',
    }

    with patch('server.routes.bitbucket_dc_proxy.httpx.AsyncClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_resp, user_resp])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        response = client.get(
            '/bitbucket-dc-proxy/oauth2/userinfo',
            headers={'Authorization': 'Bearer some_token'},
        )

    assert response.status_code == 200
    data = response.json()
    assert data['sub'] == '42'
    assert data['preferred_username'] == 'jsmith'
    assert data['name'] == 'John Smith'
    assert data['email'] == 'john@example.com'
    mock_client.get.assert_has_calls(
        [
            call(
                'https://bitbucket.test/plugins/servlet/applinks/whoami',
                headers={'Authorization': 'Bearer some_token'},
                timeout=10,
            ),
            call(
                'https://bitbucket.test/rest/api/latest/users/jsmith',
                headers={'Authorization': 'Bearer some_token'},
                timeout=10,
            ),
        ]
    )


def test_happy_path_missing_id_falls_back_to_username(client):
    whoami_resp = MagicMock()
    whoami_resp.status_code = 200
    whoami_resp.text = 'jsmith'

    user_resp = MagicMock()
    user_resp.status_code = 200
    user_resp.json.return_value = {
        'name': 'jsmith',
        'displayName': 'John Smith',
        'emailAddress': 'john@example.com',
    }

    with patch('server.routes.bitbucket_dc_proxy.httpx.AsyncClient') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_resp, user_resp])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        response = client.get(
            '/bitbucket-dc-proxy/oauth2/userinfo',
            headers={'Authorization': 'Bearer some_token'},
        )

    assert response.status_code == 200
    assert response.json()['sub'] == 'jsmith'
    mock_client.get.assert_has_calls(
        [
            call(
                'https://bitbucket.test/plugins/servlet/applinks/whoami',
                headers={'Authorization': 'Bearer some_token'},
                timeout=10,
            ),
            call(
                'https://bitbucket.test/rest/api/latest/users/jsmith',
                headers={'Authorization': 'Bearer some_token'},
                timeout=10,
            ),
        ]
    )
