from __future__ import annotations

from litellm import supports_response_schema
from pydantic import BaseModel

from openhands.core.config.condenser_config import LLMAttentionCondenserConfig
from openhands.events.action.agent import CondensationAction
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)


class ImportantEventSelection(BaseModel):
    """Utility class for the `LLMAttentionCondenser` that forces the LLM to return a list of integers."""

    ids: list[int]


class LLMAttentionCondenser(RollingCondenser):
    """Rolling condenser strategy that uses an LLM to select the most important events when condensing the history."""

    def __init__(
        self,
        llm: LLM,
        max_size: int = 100,
        keep_first: int = 1,
    ):
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if max_size < 1:
            raise ValueError(f'max_size ({keep_first}) cannot be non-positive')

        self.max_size = max_size
        self.keep_first = keep_first
        self.llm = llm

        # This condenser relies on the `response_schema` feature, which is not supported by all LLMs
        if not supports_response_schema(
            model=self.llm.config.model,
            custom_llm_provider=self.llm.config.custom_llm_provider,
        ):
            raise ValueError(
                "The LLM model must support the 'response_schema' parameter to use the LLMAttentionCondenser."
            )

        super().__init__()

    def get_condensation(self, view: View) -> Condensation:
        target_size = self.max_size // 2
        head_event_ids = [event.id for event in view.events[: self.keep_first]]

        events_from_tail = target_size - len(head_event_ids)

        message: str = """You will be given a list of actions, observations, and thoughts from a coding agent.
        Each item in the list has an identifier. Please sort the identifiers in order of how important the
        contents of the item are for the next step of the coding agent's task, from most important to least
        important."""

        response = self.llm.completion(
            messages=[
                {'content': message, 'role': 'user'},
                *[
                    {
                        'content': f'<ID>{e.id}</ID>\n<CONTENT>{e.message}</CONTENT>',
                        'role': 'user',
                    }
                    for e in view
                ],
            ],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'ImportantEventSelection',
                    'schema': ImportantEventSelection.model_json_schema(),
                },
            },
        )

        response_ids = ImportantEventSelection.model_validate_json(
            response.choices[0].message.content
        ).ids

        self.add_metadata('metrics', self.llm.metrics.get())

        # Filter out any IDs from the head and trim the results down
        response_ids = [
            response_id
            for response_id in response_ids
            if response_id not in head_event_ids
        ][:events_from_tail]

        # If the response IDs aren't _long_ enough, iterate backwards through the events and add any unfound IDs to the list.
        for event in reversed(view):
            if len(response_ids) >= events_from_tail:
                break
            if event.id not in response_ids:
                response_ids.append(event.id)

        # Now that we've found the right number of events to keep, convert this into a list of events to forget.
        event = CondensationAction(
            forgotten_event_ids=[
                event.id
                for event in view
                if event.id not in response_ids and event.id not in head_event_ids
            ],
        )

        return Condensation(action=event)

    def should_condense(self, view: View) -> bool:
        return len(view) > self.max_size or view.unhandled_condensation_request

    @classmethod
    def from_config(
        cls, config: LLMAttentionCondenserConfig, llm_registry: LLMRegistry
    ) -> LLMAttentionCondenser:
        # This condenser cannot take advantage of prompt caching. If it happens
        # to be set, we'll pay for the cache writes but never get a chance to
        # save on a read.
        llm_config = config.llm_config.model_copy()
        llm_config.caching_prompt = False

        llm = llm_registry.get_llm('condenser', llm_config)

        return LLMAttentionCondenser(
            llm=llm,
            max_size=config.max_size,
            keep_first=config.keep_first,
        )


LLMAttentionCondenser.register_config(LLMAttentionCondenserConfig)
