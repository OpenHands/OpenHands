from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM


class PipelineCondenser(CondenserBase):
    """A condenser that applies a sequence of condensers in order.

    All condensers are defined primarily by their `condense` method, which takes a
    `View` and an optional `agent_llm` parameter, returning either a new `View` or a
    `Condensation` event. That means we can chain multiple condensers together by
    passing `View`s along and exiting early if any condenser returns a `Condensation`.

    For example:

        # Use the pipeline condenser to chain multiple other condensers together
        condenser = PipelineCondenser(condensers=[
            CondenserA(...),
            CondenserB(...),
            CondenserC(...),
        ])

        result = condenser.condense(view, agent_llm=agent_llm)

        # Doing the same thing without the pipeline condenser requires more boilerplate
        # for the monadic chaining
        other_result = view

        if isinstance(other_result, View):
            other_result = CondenserA(...).condense(other_result, agent_llm=agent_llm)

        if isinstance(other_result, View):
            other_result = CondenserB(...).condense(other_result, agent_llm=agent_llm)

        if isinstance(other_result, View):
            other_result = CondenserC(...).condense(other_result, agent_llm=agent_llm)

        assert result == other_result
    """

    condensers: list[CondenserBase]
    """The list of condensers to apply in order."""

    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
        result: View | Condensation = view
        for condenser in self.condensers:
            if isinstance(result, Condensation):
                break
            result = condenser.condense(result, agent_llm=agent_llm)
        return result

    def handles_condensation_requests(self) -> bool:
        return any(
            condenser.handles_condensation_requests() for condenser in self.condensers
        )
