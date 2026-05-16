import os
from unittest.mock import patch

from openhands.app_server.sandbox.docker_sandbox_spec_service import (
    get_default_sandbox_specs as get_default_docker_sandbox_specs,
)
from openhands.app_server.sandbox.sandbox_spec_service import get_agent_server_image


class TestGetAgentServerImage:
    def test_uses_agent_server_repository_and_tag(self):
        env_vars = {
            'AGENT_SERVER_IMAGE_REPOSITORY': 'example.com/custom-agent-server',
            'AGENT_SERVER_IMAGE_TAG': 'v1-python',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            assert (
                get_agent_server_image() == 'example.com/custom-agent-server:v1-python'
            )

    def test_sandbox_runtime_container_image_legacy_override_wins(self):
        env_vars = {
            'SANDBOX_RUNTIME_CONTAINER_IMAGE': 'example.com/custom-runtime:latest',
            'AGENT_SERVER_IMAGE_REPOSITORY': 'ghcr.io/openhands/agent-server',
            'AGENT_SERVER_IMAGE_TAG': '1.19.1-python',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            assert get_agent_server_image() == 'example.com/custom-runtime:latest'

    def test_docker_spec_uses_sandbox_runtime_container_image(self):
        env_vars = {
            'SANDBOX_RUNTIME_CONTAINER_IMAGE': 'example.com/custom-runtime:latest',
            'AGENT_SERVER_IMAGE_REPOSITORY': 'ghcr.io/openhands/agent-server',
            'AGENT_SERVER_IMAGE_TAG': '1.19.1-python',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            specs = get_default_docker_sandbox_specs()

            assert len(specs) == 1
            assert specs[0].id == 'example.com/custom-runtime:latest'
