import ssl

_verify_certificates: bool = True


def httpx_verify_option() -> ssl.SSLContext | bool:
    """Return the verify option to pass when creating an HTTPX client."""

    return ssl.create_default_context() if _verify_certificates else False
