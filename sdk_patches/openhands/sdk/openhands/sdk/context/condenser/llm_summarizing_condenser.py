import os
from collections.abc import Sequence
from enum import Enum

from pydantic import Field, model_validator

from openhands.sdk.context.condenser.base import RollingCondenser
from openhands.sdk.context.condenser.utils import (
    get_suffix_length_for_token_reduction,
    get_total_token_count,
)
from openhands.sdk.context.prompts import render_template
from openhands.sdk.context.view import View
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.observability.laminar import observe


class Reason(Enum):
    """Reasons for condensation."""

    REQUEST = "request"
    TOKENS = "tokens"
    EVENTS = "events"


class LLMSummarizingCondenser(RollingCondenser):
    """LLM-based condenser that summarizes forgotten events.

    Uses an independent LLM (stored in the `llm` attribute) for generating summaries
    of forgotten events. The optional `agent_llm` parameter passed to condense() is
    the LLM used by the agent for token counting purposes, and you should not assume
    it is the same as the one defined in this condenser.
    """

    llm: LLM
    max_size: int = Field(default=240, gt=0)
    max_tokens: int | None = None
    keep_first: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_keep_first_vs_max_size(self):
        events_from_tail = self.max_size // 2 - self.keep_first - 1
        if events_from_tail <= 0:
            raise ValueError(
                "keep_first must be less than max_size // 2 to leave room for "
                "condensation"
            )
        return self

    def handles_condensation_requests(self) -> bool:
        return True

    def get_condensation_reasons(
        self, view: View, agent_llm: LLM | None = None
    ) -> set[Reason]:
        """Determine the reasons why the view should be condensed.

        Args:
            view: The current view to evaluate.
            agent_llm: The LLM used by the agent. Required if token counting is needed.

        Returns:
            A set of Reason enums indicating why condensation is needed.
        """
        reasons = set()

        # Reason 1: Unhandled condensation request. The view handles the detection of
        # these requests while processing the event stream.
        if view.unhandled_condensation_request:
            reasons.add(Reason.REQUEST)

        # Reason 2: Token limit is provided and exceeded.
        if self.max_tokens and agent_llm:
            total_tokens = get_total_token_count(view.events, agent_llm)
            if total_tokens > self.max_tokens:
                reasons.add(Reason.TOKENS)

        # Reason 3: View exceeds maximum size in number of events.
        if len(view) > self.max_size:
            reasons.add(Reason.EVENTS)

        return reasons

    def should_condense(self, view: View, agent_llm: LLM | None = None) -> bool:
        reasons = self.get_condensation_reasons(view, agent_llm)
        return reasons != set()

    def _get_summary_event_content(self, view: View) -> str:
        """Extract the text content from the summary event in the view, if any.

        If there is no summary event or it does not contain text content, returns an
        empty string.
        """
        summary_event_content: str = ""

        summary_event = view.summary_event
        if isinstance(summary_event, MessageEvent):
            message_content = summary_event.llm_message.content[0]
            if isinstance(message_content, TextContent):
                summary_event_content = message_content.text

        return summary_event_content

    def _generate_condensation(
        self,
        summary_event_content: str,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
    ) -> Condensation:
        """Generate a condensation by using the condenser's LLM to summarize forgotten
        events.

        Args:
            summary_event_content: The content of the previous summary event.
            forgotten_events: The list of events to be summarized.
            summary_offset: The index where the summary event should be inserted.

        Returns:
            Condensation: The generated condensation object.

        Raises:
            ValueError: If forgotten_events is empty (0 events to condense).
        """
        if len(forgotten_events) == 0:
            raise ValueError(
                "Cannot condense 0 events. This typically occurs when a tool loop "
                "spans almost the entire view, leaving no valid range for forgetting "
                "events. Consider adjusting keep_first or max_size parameters."
            )

        # Convert events to strings for the template
        event_strings = [str(forgotten_event) for forgotten_event in forgotten_events]

        prompt = render_template(
            os.path.join(os.path.dirname(__file__), "prompts"),
            "summarizing_prompt.j2",
            previous_summary=summary_event_content,
            events=event_strings,
        )

        messages = [Message(role="user", content=[TextContent(text=prompt)])]

        # Do not pass extra_body explicitly. The LLM handles forwarding
        # litellm_extra_body only when it is non-empty.
        llm_response = self.llm.completion(
            messages=messages,
        )
        # Extract summary from the LLMResponse message
        summary = None
        if llm_response.message.content:
            first_content = llm_response.message.content[0]
            if isinstance(first_content, TextContent):
                summary = first_content.text

        return Condensation(
            forgotten_event_ids=[event.id for event in forgotten_events],
            summary=summary,
            summary_offset=summary_offset,
            llm_response_id=llm_response.id,
        )

    def _get_forgotten_events(
        self, view: View, agent_llm: LLM | None = None
    ) -> tuple[Sequence[LLMConvertibleEvent], int]:
        """Identify events to be forgotten and the summary offset.

        Relies on the condensation reasons to determine how many events we need to drop
        in order to maintain our resource constraints. Uses manipulation indices to
        ensure forgetting ranges respect atomic unit boundaries.

        Args:
            view: The current view from which to identify forgotten events.
            agent_llm: The LLM used by the agent, required for token-based calculations.

        Returns:
            A tuple of (events to forget, summary_offset).
        """
        reasons = self.get_condensation_reasons(view, agent_llm=agent_llm)
        assert reasons != set(), "No condensation reasons found."

        suffix_events_to_keep: set[int] = set()

        if Reason.REQUEST in reasons:
            target_size = len(view) // 2
            suffix_events_to_keep.add(target_size - self.keep_first - 1)

        if Reason.EVENTS in reasons:
            target_size = self.max_size // 2
            suffix_events_to_keep.add(target_size - self.keep_first - 1)

        if Reason.TOKENS in reasons:
            # Compute the number of tokens we need to eliminate to be under half the
            # max_tokens value. We know max_tokens and the agent LLM are not None here
            # because we can't have Reason.TOKENS without them.
            assert self.max_tokens is not None
            assert agent_llm is not None

            total_tokens = get_total_token_count(view.events, agent_llm)
            tokens_to_reduce = total_tokens - (self.max_tokens // 2)

            suffix_events_to_keep.add(
                get_suffix_length_for_token_reduction(
                    events=view.events[self.keep_first :],
                    llm=agent_llm,
                    token_reduction=tokens_to_reduce,
                )
            )

        # We might have multiple reasons to condense, so pick the strictest condensation
        # to ensure all resource constraints are met.
        events_from_tail = min(suffix_events_to_keep)

        # Calculate naive forgetting end (without considering atomic boundaries)
        naive_end = len(view) - events_from_tail

        # Find actual forgetting_start: smallest manipulation index > keep_first
        forgetting_start = view.find_next_manipulation_index(
            self.keep_first, strict=True
        )

        # Find actual forgetting_end: smallest manipulation index >= naive_end
        forgetting_end = view.find_next_manipulation_index(naive_end, strict=False)

        # Extract events to forget using boundary-aware indices
        forgotten_events = view[forgetting_start:forgetting_end]

        # Summary offset is the same as forgetting_start
        return forgotten_events, forgetting_start

    @observe(ignore_inputs=["view", "agent_llm"])
    def get_condensation(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation:
        # The condensation is dependent on the events we want to drop and the previous
        # summary.
        summary_event_content = self._get_summary_event_content(view)
        forgotten_events, summary_offset = self._get_forgotten_events(
            view, agent_llm=agent_llm
        )

        return self._generate_condensation(
            summary_event_content=summary_event_content,
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
        )
