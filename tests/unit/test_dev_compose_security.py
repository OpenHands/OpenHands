from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_COMPOSE = REPO_ROOT / 'containers/dev/compose.yml'
DOCKER_SOCKET_OVERRIDE = REPO_ROOT / 'containers/dev/compose.docker-socket.yml'


def test_dev_compose_does_not_mount_host_docker_socket_by_default():
    content = DEV_COMPOSE.read_text()

    assert '/var/run/docker.sock' not in content
    assert 'privileged: true' not in content


def test_docker_socket_access_is_explicit_override_only():
    content = DOCKER_SOCKET_OVERRIDE.read_text()

    assert '/var/run/docker.sock:/var/run/docker.sock' in content
    assert 'privileged: true' in content
