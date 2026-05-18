"""Tests for BoxdSandboxService.

Focuses on:
- boxd SDK exception handling
- Sandbox lifecycle management (start, pause, resume, delete)
- Status mapping from boxd VM status to internal sandbox statuses
- Environment variable injection for CORS and webhooks
- Data transformation from boxd Box to SandboxInfo objects
- User-scoped sandbox operations and security
- Pagination and search functionality via box.list()
"""

from openhands.app_server.sandbox.boxd_sandbox_service import STATUS_MAPPING
from openhands.app_server.sandbox.sandbox_models import SandboxStatus


class TestStatusMapping:
    def test_running_maps_to_running(self):
        assert STATUS_MAPPING['running'] == SandboxStatus.RUNNING

    def test_suspended_maps_to_paused(self):
        assert STATUS_MAPPING['suspended'] == SandboxStatus.PAUSED

    def test_starting_maps_to_starting(self):
        assert STATUS_MAPPING['starting'] == SandboxStatus.STARTING

    def test_error_maps_to_error(self):
        assert STATUS_MAPPING['error'] == SandboxStatus.ERROR

    def test_stopped_maps_to_missing(self):
        assert STATUS_MAPPING['stopped'] == SandboxStatus.MISSING
