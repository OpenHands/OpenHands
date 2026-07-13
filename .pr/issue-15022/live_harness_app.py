from __future__ import annotations

import logging

from live_services import close_config_resources, patch_local_services

logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

from saas_server import base_app as app  # noqa: E402

patch_local_services()
app.router.lifespan_context = lambda _app: close_config_resources()
