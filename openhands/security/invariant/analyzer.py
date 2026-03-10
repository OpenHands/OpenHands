# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import re
import uuid
from typing import Any

import docker

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action, ActionSecurityRisk
from openhands.runtime.utils import find_available_tcp_port
from openhands.security.analyzer import SecurityAnalyzer
from openhands.security.invariant.client import InvariantClient
from openhands.security.invariant.parser import TraceElement, parse_element


class InvariantAnalyzer(SecurityAnalyzer):
    """Security analyzer based on Invariant - purely analytical."""

    trace: list[TraceElement]
    input: list[dict[str, Any]]
    container_name: str = 'openhands-invariant-server'
    image_name: str = 'ghcr.io/invariantlabs-ai/server:openhands'
    api_host: str = 'http://localhost'
    timeout: int = 180

    def __init__(
        self,
        policy: str | None = None,
        sid: str | None = None,
    ) -> None:
        """Initializes a new instance of the InvariantAnalyzer class."""
        super().__init__()
        self.trace = []
        self.input = []
        if sid is None:
            self.sid = str(uuid.uuid4())

        try:
            self.docker_client = docker.from_env()
        except Exception as ex:
            logger.exception(
                'Error creating Invariant Security Analyzer container. Please check that Docker is running or disable the Security Analyzer in settings.',
                exc_info=False,
            )
            raise ex
        running_containers = self.docker_client.containers.list(
            filters={'name': self.container_name}
        )
        if not running_containers:
            all_containers = self.docker_client.containers.list(
                all=True, filters={'name': self.container_name}
            )
            if all_containers:
                self.container = all_containers[0]
                all_containers[0].start()
            else:
                self.api_port = find_available_tcp_port()
                self.container = self.docker_client.containers.run(
                    self.image_name,
                    name=self.container_name,
                    platform='linux/amd64',
                    ports={'8000/tcp': self.api_port},
                    detach=True,
                )
        else:
            self.container = running_containers[0]

        elapsed = 0
        while self.container.status != 'running':
            self.container = self.docker_client.containers.get(self.container_name)
            elapsed += 1
            logger.debug(
                f'waiting for container to start: {elapsed}, container status: {self.container.status}'
            )
            if elapsed > self.timeout:
                break

        self.api_port = int(
            self.container.attrs['NetworkSettings']['Ports']['8000/tcp'][0]['HostPort']
        )

        self.api_server = f'{self.api_host}:{self.api_port}'
        self.client = InvariantClient(self.api_server, self.sid)
        if policy is None:
            policy, _ = self.client.Policy.get_template()
            if policy is None:
                policy = ''
        self.monitor = self.client.Monitor.from_string(policy)

    async def close(self) -> None:
        self.container.stop()

    def get_risk(self, results: list[str]) -> ActionSecurityRisk:
        mapping = {
            'high': ActionSecurityRisk.HIGH,
            'medium': ActionSecurityRisk.MEDIUM,
            'low': ActionSecurityRisk.LOW,
        }
        regex = r'(?<=risk=)\w+'
        risks: list[ActionSecurityRisk] = []
        for result in results:
            m = re.search(regex, result)
            if m and m.group() in mapping:
                risks.append(mapping[m.group()])

        if risks:
            return max(risks)

        return ActionSecurityRisk.LOW

    async def security_risk(self, action: Action) -> ActionSecurityRisk:
        logger.debug('Calling security_risk on InvariantAnalyzer')
        new_elements = parse_element(self.trace, action)
        input_data = [e.model_dump(exclude_none=True) for e in new_elements]
        self.trace.extend(new_elements)
        check_result = self.monitor.check(self.input, input_data)
        self.input.extend(input_data)
        risk = ActionSecurityRisk.UNKNOWN

        # Process check_result
        result, err = check_result
        if err:
            logger.warning(f'Error checking policy: {err}')
            return risk

        return self.get_risk(result)
