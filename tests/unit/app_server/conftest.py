import pytest


@pytest.fixture(scope='session')
def base_url() -> str:
    """Provide a stable base_url fixture for pytest-base-url plugin in unit tests."""
    return 'http://localhost'
