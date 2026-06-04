from urllib.parse import urlparse, urlunparse

from openhands.app_server.utils.environment import is_running_in_docker


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
        # Rebuild the netloc from its components so that only the host is
        # changed. A naive str.replace on the raw netloc would corrupt
        # userinfo that happens to contain 'localhost' (e.g. a password equal
        # to 'localhost') and would miss a differently-cased host such as
        # 'LOCALHOST', which urlparse normalizes when exposing parsed.hostname.
        netloc = replacement
        if parsed.port is not None:
            netloc = f'{netloc}:{parsed.port}'
        if parsed.username is not None:
            userinfo = parsed.username
            if parsed.password is not None:
                userinfo = f'{userinfo}:{parsed.password}'
            netloc = f'{userinfo}@{netloc}'
        return urlunparse(parsed._replace(netloc=netloc))
    return url
