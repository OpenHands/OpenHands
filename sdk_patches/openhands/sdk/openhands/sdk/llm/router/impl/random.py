import random

from openhands.sdk.llm.message import Message
from openhands.sdk.llm.router.base import RouterLLM
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class RandomRouter(RouterLLM):
    """
    A simple implementation of RouterLLM that randomly selects an LLM from
    llms_for_routing for each completion request.
    """

    router_name: str = "random_router"

    def select_llm(self, messages: list[Message]) -> str:  # noqa: ARG002
        selected_llm_name = random.choice(list(self.llms_for_routing.keys()))
        logger.info(f"Randomly selected LLM: {selected_llm_name}")
        return selected_llm_name
