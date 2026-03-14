import pytest

from openhands.utils import environment
from openhands.utils.environment import StorageProvider, get_storage_provider


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


class TestGetStorageProvider:
    """Tests for get_storage_provider function."""

    def test_aws_from_shared_event_storage_provider(self, monkeypatch):
        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'aws')
        monkeypatch.delenv('FILE_STORE', raising=False)
        assert get_storage_provider() == StorageProvider.AWS

    def test_gcp_from_shared_event_storage_provider(self, monkeypatch):
        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'gcp')
        monkeypatch.delenv('FILE_STORE', raising=False)
        assert get_storage_provider() == StorageProvider.GCP

    def test_google_cloud_from_shared_event_storage_provider(self, monkeypatch):
        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'google_cloud')
        monkeypatch.delenv('FILE_STORE', raising=False)
        assert get_storage_provider() == StorageProvider.GCP

    def test_fallback_to_file_store_google_cloud(self, monkeypatch):
        monkeypatch.delenv('SHARED_EVENT_STORAGE_PROVIDER', raising=False)
        monkeypatch.setenv('FILE_STORE', 'google_cloud')
        assert get_storage_provider() == StorageProvider.GCP

    def test_filesystem_when_no_provider_set(self, monkeypatch):
        monkeypatch.delenv('SHARED_EVENT_STORAGE_PROVIDER', raising=False)
        monkeypatch.delenv('FILE_STORE', raising=False)
        assert get_storage_provider() == StorageProvider.FILESYSTEM

    def test_filesystem_for_unknown_provider(self, monkeypatch):
        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'unknown')
        assert get_storage_provider() == StorageProvider.FILESYSTEM

    def test_shared_event_storage_provider_takes_precedence(self, monkeypatch):
        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'aws')
        monkeypatch.setenv('FILE_STORE', 'google_cloud')
        assert get_storage_provider() == StorageProvider.AWS

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'AWS')
        assert get_storage_provider() == StorageProvider.AWS

        monkeypatch.setenv('SHARED_EVENT_STORAGE_PROVIDER', 'GCP')
        assert get_storage_provider() == StorageProvider.GCP
