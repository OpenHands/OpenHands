from ipaddress import ip_address
from urllib.parse import urlparse, urlunparse

from openhands.app_server.utils.environment import is_running_in_docker


def replace_localhost_hostname(
    url: str, replacement: str = 'host.docker.internal'
) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return url
    is_local = hostname.lower() == 'localhost'
    try:
        address = ip_address(hostname)
        is_local = is_local or address.is_loopback or address.is_unspecified
    except ValueError:
        pass
    if not is_local:
        return url
    userinfo, separator, _ = parsed.netloc.rpartition('@')
    netloc = replacement
    if parsed.port is not None:
        netloc = f'{netloc}:{parsed.port}'
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
