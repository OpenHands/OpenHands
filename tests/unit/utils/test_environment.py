import pytest

from openhands.utils import environment


@pytest.fixture(autouse=True)
def clear_docker_cache():
    if hasattr(environment.is_running_in_docker, 'cache_clear'):
        environment.is_running_in_docker.cache_clear()
    yield
    if hasattr(environment.is_running_in_docker, 'cache_clear'):
        environment.is_running_in_docker.cache_clear()


def test_get_effective_base_url_lemonade_in_docker(monkeypatch):
    monkeypatch.setattr(environment, 'is_running_in_docker', lambda: True)
    result = environment.get_effective_llm_base_url('lemonade/example', None)
    assert result == environment.LEMONADE_DOCKER_BASE_URL


def test_get_effective_base_url_lemonade_outside_docker(monkeypatch):
    monkeypatch.setattr(environment, 'is_running_in_docker', lambda: False)
    base_url = 'http://localhost:8000/api/v1/'
    result = environment.get_effective_llm_base_url('lemonade/example', base_url)
    assert result == base_url


def test_get_effective_base_url_non_lemonade(monkeypatch):
    monkeypatch.setattr(environment, 'is_running_in_docker', lambda: True)
    base_url = 'https://api.example.com'
    result = environment.get_effective_llm_base_url('openai/gpt-4', base_url)
    assert result == base_url
