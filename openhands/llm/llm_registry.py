# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import copy
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


class RegistryEvent(BaseModel):
    llm: LLM
    service_id: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class LLMRegistry:
    def __init__(
        self,
        config: OpenHandsConfig,
        agent_cls: str | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ):
        self.registry_id = str(uuid4())
        self.config = copy.deepcopy(config)
        self.retry_listner = retry_listener
        self.agent_to_llm_config = self.config.get_agent_to_llm_config_map()
        self.service_to_llm: dict[str, LLM] = {}
        self.subscriber: Callable[[Any], None] | None = None

        selected_agent_cls = self.config.default_agent
        if agent_cls:
            selected_agent_cls = agent_cls

        agent_name = selected_agent_cls if selected_agent_cls is not None else 'agent'
        llm_config = self.config.get_llm_config_from_agent(agent_name)
        self.active_agent_llm: LLM = self.get_llm('agent', llm_config)

    def _create_new_llm(
        self, service_id: str, config: LLMConfig, with_listener: bool = True
    ) -> LLM:
        if with_listener:
            llm = LLM(
                service_id=service_id, config=config, retry_listener=self.retry_listner
            )
        else:
            llm = LLM(service_id=service_id, config=config)
        self.service_to_llm[service_id] = llm
        self.notify(RegistryEvent(llm=llm, service_id=service_id))
        return llm

    def request_extraneous_completion(
        self, service_id: str, llm_config: LLMConfig, messages: list[dict[str, str]]
    ) -> str:
        logger.info(f'extraneous completion: {service_id}')
        if service_id not in self.service_to_llm:
            self._create_new_llm(
                config=llm_config, service_id=service_id, with_listener=False
            )

        llm = self.service_to_llm[service_id]
        response = llm.completion(messages=messages)
        return response.choices[0].message.content.strip()

    def get_llm_from_agent_config(self, service_id: str, agent_config: AgentConfig):
        llm_config = self.config.get_llm_config_from_agent_config(agent_config)
        if service_id in self.service_to_llm:
            if self.service_to_llm[service_id].config != llm_config:
                # TODO: update llm config internally
                # Done when agent delegates has different config, we should reuse the existing LLM
                pass
            return self.service_to_llm[service_id]

        return self._create_new_llm(config=llm_config, service_id=service_id)

    def get_llm(
        self,
        service_id: str,
        config: LLMConfig | None = None,
    ):
        logger.info(
            f'[LLM registry {self.registry_id}]: Registering service for {service_id}'
        )

        # Attempting to switch configs for existing LLM
        if (
            service_id in self.service_to_llm
            and self.service_to_llm[service_id].config != config
        ):
            raise ValueError(
                f'Requesting same service ID {service_id} with different config, use a new service ID'
            )

        if service_id in self.service_to_llm:
            return self.service_to_llm[service_id]

        if not config:
            raise ValueError('Requesting new LLM without specifying LLM config')

        return self._create_new_llm(config=config, service_id=service_id)

    def get_active_llm(self) -> LLM:
        return self.active_agent_llm

    def get_router(self, agent_config: AgentConfig) -> 'LLM':
        """
        Get a router instance that inherits from LLM.
        """
        # Import here to avoid circular imports
        from openhands.llm.router import RouterLLM

        router_name = agent_config.model_routing.router_name

        if router_name == 'noop_router':
            # Return the main LLM directly (no routing)
            return self.get_llm_from_agent_config('agent', agent_config)

        return RouterLLM.from_config(
            agent_config=agent_config,
            llm_registry=self,
            retry_listener=self.retry_listner,
        )

    def subscribe(self, callback: Callable[[RegistryEvent], None]) -> None:
        self.subscriber = callback

        # Subscriptions happen after default llm is initialized
        # Notify service of this llm
        self.notify(
            RegistryEvent(
                llm=self.active_agent_llm, service_id=self.active_agent_llm.service_id
            )
        )

    def notify(self, event: RegistryEvent):
        if self.subscriber:
            try:
                self.subscriber(event)
            except Exception as e:
                logger.warning(f'Failed to emit event: {e}')
