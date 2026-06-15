import os
from typing import Literal

DeploymentMode = Literal['cloud', 'self_hosted']


# Honors an explicit OH_DEPLOYMENT_MODE; the OH_WEB_HOST/WEB_HOST heuristic is the fallback.
def get_deployment_mode() -> DeploymentMode | None:
    """Get deployment mode.

    Honors an explicit OH_DEPLOYMENT_MODE ('cloud' | 'self_hosted'); otherwise
    infers from OH_WEB_HOST/WEB_HOST (managed domain -> 'cloud'), or None if unset.
    """
    explicit = os.getenv('OH_DEPLOYMENT_MODE', '').strip().lower()
    if explicit == 'cloud':
        return 'cloud'
    if explicit == 'self_hosted':
        return 'self_hosted'
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
