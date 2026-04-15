import os
from typing import Literal

DeploymentMode = Literal['cloud', 'self_hosted']


# This can be removed / replaced when a DeploymentMode (or similar) env var is created.
def get_deployment_mode() -> DeploymentMode | None:
    """Get deployment mode based on OH_WEB_HOST environment variable.

    Returns:
        'cloud' for All-Hands managed infrastructure (app.all-hands.dev, etc.)
        'self_hosted' for enterprise self-hosted deployments (customer domains)
        None if WEB_HOST is not set
    """
    web_host = os.getenv('OH_WEB_HOST', os.getenv('WEB_HOST', '')).strip()
    if not web_host:
        return None
    if (
        web_host == 'app.all-hands.dev'
        or web_host == 'app.openhands.ai'
        or web_host.endswith('.all-hands.dev')
        or web_host.endswith('.openhands.ai')
    ):
        return 'cloud'
    return 'self_hosted'
