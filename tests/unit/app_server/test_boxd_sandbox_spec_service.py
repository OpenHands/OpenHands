"""Tests for BoxdSandboxSpecServiceInjector and the default boxd sandbox spec."""

import pytest

from openhands.app_server.sandbox.boxd_sandbox_spec_service import (
    BoxdSandboxSpecServiceInjector,
    get_default_boxd_sandbox_specs,
)
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.services.injector import InjectorState


class TestBoxdSandboxSpecDefaults:
    def test_default_specs_returns_one_spec(self):
        specs = get_default_boxd_sandbox_specs()
        assert len(specs) == 1

    def test_default_spec_uses_agent_server_image(self):
        specs = get_default_boxd_sandbox_specs()
        # The spec id IS the image tag — agent-server is keyed by image.
        assert 'agent-server' in specs[0].id

    def test_default_spec_has_agent_server_command(self):
        specs = get_default_boxd_sandbox_specs()
        cmd = specs[0].command
        assert cmd is not None
        assert any('openhands-agent-server' in part for part in cmd)
        assert '--port' in cmd
        assert '60000' in cmd

    def test_default_spec_working_dir_is_project(self):
        specs = get_default_boxd_sandbox_specs()
        assert specs[0].working_dir == '/workspace/project'

    def test_default_spec_initial_env_includes_log_json(self):
        specs = get_default_boxd_sandbox_specs()
        assert specs[0].initial_env.get('LOG_JSON') == 'true'


class TestBoxdSandboxSpecServiceInjector:
    @pytest.mark.asyncio
    async def test_inject_yields_preset_service(self):
        injector = BoxdSandboxSpecServiceInjector()
        state = InjectorState()
        async for service in injector.inject(state, request=None):
            assert isinstance(service, PresetSandboxSpecService)
            page = await service.search_sandbox_specs()
            assert len(page.items) == 1
            break
