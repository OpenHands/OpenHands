# DEPRECATED: This module is part of the deprecated 'openhands.runtime' package.
# It will be removed on April 1, 2025. Please migrate to the OpenHands SDK:
# https://github.com/All-Hands-AI/openhands-sdk
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
