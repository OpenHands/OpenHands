from urllib.parse import urlparse, urlunparse

from openhands.app_server.utils.environment import is_running_in_docker


def replace_localhost_hostname(
    url: str, replacement: str = 'host.docker.internal'
) -> str:
    parsed = urlparse(url)
    if parsed.hostname != 'localhost':
        return url
    userinfo, separator, host_and_port = parsed.netloc.rpartition('@')
    if not separator:
        host_and_port = parsed.netloc
    _, port_separator, port = host_and_port.partition(':')
    netloc = f'{replacement}{port_separator}{port}'
    if separator:
        netloc = f'{userinfo}@{netloc}'
    return urlunparse(parsed._replace(netloc=netloc))


def replace_localhost_hostname_for_docker(
    url: str, replacement: str = 'host.docker.internal'
) -> str:
    """Replace localhost hostname in URL with the specified replacement when running in Docker.

    This function only performs the replacement when the code is running inside a Docker
    container. When not running in Docker, it returns the original URL unchanged.

    Only replaces the hostname if it's exactly 'localhost', preserving all other
    parts of the URL including port, path, query parameters, etc.

    Args:
        url: The URL to process
        replacement: The hostname to replace localhost with (default: 'host.docker.internal')

    Returns:
        URL with localhost hostname replaced if running in Docker and hostname is localhost,
        otherwise returns the original URL unchanged
    """
    if not is_running_in_docker():
        return url
    return replace_localhost_hostname(url, replacement)
