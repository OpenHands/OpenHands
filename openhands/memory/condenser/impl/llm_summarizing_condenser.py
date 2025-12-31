from __future__ import annotations

from openhands.core.config.condenser_config import LLMSummarizingCondenserConfig
from openhands.core.message import Message, TextContent
from openhands.events.action.agent import CondensationAction
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)


class LLMSummarizingCondenser(RollingCondenser):
    """A condenser that summarizes forgotten events.

    Maintains a condensed history and forgets old events when it grows too large,
    keeping a special summarization event after the prefix that summarizes all previous summarizations
    and newly forgotten events.
    """

    def __init__(
        self,
        llm: LLM,
        max_size: int = 100,
        keep_first: int = 1,
        max_event_length: int = 10_000,
    ):
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if max_size < 1:
            raise ValueError(f'max_size ({max_size}) cannot be non-positive')

        self.max_size = max_size
        self.keep_first = keep_first
        self.max_event_length = max_event_length
        self.llm = llm

        super().__init__()

    def _truncate(self, content: str) -> str:
        """Truncate the content to fit within the specified maximum event length."""
        return truncate_content(content, max_chars=self.max_event_length)

    def get_condensation(self, view: View) -> Condensation:
        head = view[: self.keep_first]
        target_size = self.max_size // 2
        # Number of events to keep from the tail -- target size, minus however many
        # prefix events from the head, minus one for the summarization event
        events_from_tail = target_size - len(head) - 1

        summary_event = (
            view[self.keep_first]
            if isinstance(view[self.keep_first], AgentCondensationObservation)
            else AgentCondensationObservation('No events summarized')
        )

        # Identify events to be forgotten (those not in head or tail)
        forgotten_events = []
        for event in view[self.keep_first : -events_from_tail]:
            if not isinstance(event, AgentCondensationObservation):
                forgotten_events.append(event)

        # Construct prompt for summarization
        prompt = """You are maintaining a context-aware state summary for an interactive agent.
You will be given a list of events corresponding to actions taken by the agent, and the most recent previous summary if one exists.
If the events being summarized contain ANY task-tracking, you MUST include a TASK_TRACKING section to maintain continuity.
When referencing tasks make sure to preserve exact task IDs and statuses.

Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

TASK_TRACKING: {Active tasks, their IDs and statuses - PRESERVE TASK IDs}

COMPLETED: (Tasks completed so far, with brief results)
PENDING: (Tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant state)

For code-specific tasks, also include:
CODE_STATE: {File paths, function signatures, data structures}
TESTS: {Failing cases, error messages, outputs}
CHANGES: {Code edits, variable updates}
DEPS: {Dependencies, imports, external calls}
VERSION_CONTROL_STATUS: {Repository state, current branch, PR status, commit history}

PRIORITIZE:
1. Adapt tracking format to match the actual task type
2. Capture key user requirements and goals
3. Distinguish between completed and pending tasks
4. Keep all sections concise and relevant

SKIP: Tracking irrelevant details for the current task type

Example formats:

For code tasks:
USER_CONTEXT: Fix FITS card float representation issue
COMPLETED: Modified mod_float() in card.py, all tests passing
PENDING: Create PR, update documentation
CODE_STATE: mod_float() in card.py updated
TESTS: test_format() passed
CHANGES: str(val) replaces f"{val:.16G}"
DEPS: None modified
VERSION_CONTROL_STATUS: Branch: fix-float-precision, Latest commit: a1b2c3d

For other tasks:
USER_CONTEXT: Write 20 haikus based on coin flip results
COMPLETED: 15 haikus written for results [T,H,T,H,T,H,T,T,H,T,H,T,H,T,H]
PENDING: 5 more haikus needed
CURRENT_STATE: Last flip: Heads, Haiku count: 15/20"""

        prompt += '\n\n'

        # Add the previous summary if it exists. We'll always have a summary
        # event, but the types aren't precise enought to guarantee that it has a
        # message attribute.
        summary_event_content = self._truncate(
            summary_event.message if summary_event.message else ''
        )
        prompt += f'<PREVIOUS SUMMARY>\n{summary_event_content}\n</PREVIOUS SUMMARY>\n'

        prompt += '\n\n'

        # Add all events that are being forgotten. We use the string
        # representation defined by the event, and truncate it if necessary.
        for forgotten_event in forgotten_events:
            event_content = self._truncate(str(forgotten_event))
            prompt += f'<EVENT id={forgotten_event.id}>\n{event_content}\n</EVENT>\n'

        prompt += 'Now summarize the events using the rules above.'

        messages = [Message(role='user', content=[TextContent(text=prompt)])]

        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
            extra_body={'metadata': self.llm_metadata},
        )
        summary = response.choices[0].message.content

        self.add_metadata('response', response.model_dump())
        self.add_metadata('metrics', self.llm.metrics.get())

        return Condensation(
            action=CondensationAction(
                forgotten_events_start_id=min(event.id for event in forgotten_events),
                forgotten_events_end_id=max(event.id for event in forgotten_events),
                summary=summary,
                summary_offset=self.keep_first,
            )
        )

    def should_condense(self, view: View) -> bool:
        return len(view) > self.max_size or view.unhandled_condensation_request

    @classmethod
    def from_config(
        cls, config: LLMSummarizingCondenserConfig, llm_registry: LLMRegistry
    ) -> LLMSummarizingCondenser:
        # This condenser cannot take advantage of prompt caching. If it happens
        # to be set, we'll pay for the cache writes but never get a chance to
        # save on a read.
        llm_config = config.llm_config.model_copy()
        llm_config.caching_prompt = False
        llm = llm_registry.get_llm('condenser', llm_config)

        return LLMSummarizingCondenser(
            llm=llm,
            max_size=config.max_size,
            keep_first=config.keep_first,
            max_event_length=config.max_event_length,
        )


LLMSummarizingCondenser.register_config(LLMSummarizingCondenserConfig)
