from urllib.parse import urlparse, urlunparse

from openhands.utils.environment import is_running_in_docker


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
    parsed = urlparse(url)
    if parsed.hostname == 'localhost':
        # When running in Docker, check if we're in the bridge network
        # If so, use the bridge gateway IP (172.17.0.1) instead of host.docker.internal
        # This is because containers in the bridge network can't resolve host.docker.internal
        import os
        # Check if we're in the bridge network by checking if host.docker.internal is resolvable
        # If not, use the bridge gateway IP
        if os.environ.get('USE_BRIDGE_GATEWAY', 'false').lower() == 'true':
            replacement = '172.17.0.1'
        # Replace only the hostname part, preserving port and everything else
        netloc = parsed.netloc.replace('localhost', replacement, 1)
        return urlunparse(parsed._replace(netloc=netloc))
    return url
