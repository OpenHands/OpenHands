import pytest

from openhands.app_server.app import _should_serve_frontend


@pytest.mark.parametrize(
    ('serve_frontend', 'expected'),
    [(None, True), ('true', True), ('1', True), ('false', False), ('0', False)],
)
def test_serve_frontend_env(serve_frontend, expected, monkeypatch):
    if serve_frontend is None:
        monkeypatch.delenv('SERVE_FRONTEND', raising=False)
    else:
        monkeypatch.setenv('SERVE_FRONTEND', serve_frontend)

    assert _should_serve_frontend() is expected
