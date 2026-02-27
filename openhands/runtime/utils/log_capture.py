# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import io
import logging
from contextlib import asynccontextmanager


@asynccontextmanager
async def capture_logs(logger_name, level=logging.ERROR):
    logger = logging.getLogger(logger_name)

    # Store original handlers and level
    original_handlers = logger.handlers[:]
    original_level = logger.level

    # Set up capture
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(level)

    logger.handlers = [handler]
    logger.setLevel(level)

    try:
        yield log_capture
    finally:
        # Restore original configuration
        logger.handlers = original_handlers
        logger.setLevel(original_level)
