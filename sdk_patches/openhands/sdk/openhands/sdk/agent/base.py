import os
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from openhands.sdk.context.agent_context import AgentContext
from openhands.sdk.context.condenser import CondenserBase, LLMSummarizingCondenser
from openhands.sdk.context.prompts.prompt import render_template
from openhands.sdk.llm import LLM
from openhands.sdk.llm.utils.model_prompt_spec import get_model_prompt_spec
from openhands.sdk.logger import get_logger
from openhands.sdk.mcp import create_mcp_tools
from openhands.sdk.tool import BUILT_IN_TOOLS, Tool, ToolDefinition, resolve_tool
from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.utils.pydantic_diff import pretty_pydantic_diff


if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState, LocalConversation
    from openhands.sdk.conversation.types import (
        ConversationCallbackType,
        ConversationTokenCallbackType,
    )


logger = get_logger(__name__)


class AgentBase(DiscriminatedUnionMixin, ABC):
    """Abstract base class for OpenHands agents.

    Agents are stateless and should be fully defined by their configuration.
    This base class provides the common interface and functionality that all
    agent implementations must follow.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    llm: LLM = Field(
        ...,
        description="LLM configuration for the agent.",
        examples=[
            {
                "model": "litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
                "base_url": "https://llm-proxy.eval.all-hands.dev",
                "api_key": "your_api_key_here",
            }
        ],
    )
    tools: list[Tool] = Field(
        default_factory=list,
        description="List of tools to initialize for the agent.",
        examples=[
            {"name": "TerminalTool", "params": {}},
            {"name": "FileEditorTool", "params": {}},
            {
                "name": "TaskTrackerTool",
                "params": {},
            },
        ],
    )
    mcp_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional MCP configuration dictionary to create MCP tools.",
        examples=[
            {"mcpServers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}}
        ],
    )
    filter_tools_regex: str | None = Field(
        default=None,
        description="Optional regex to filter the tools available to the agent by name."
        " This is applied after any tools provided in `tools` and any MCP tools are"
        " added.",
        examples=["^(?!repomix)(.*)|^repomix.*pack_codebase.*$"],
    )
    agent_context: AgentContext | None = Field(
        default=None,
        description="Optional AgentContext to initialize "
        "the agent with specific context.",
        examples=[
            {
                "skills": [
                    {
                        "name": "repo.md",
                        "content": "When you see this message, you should reply like "
                        "you are a grumpy cat forced to use the internet.",
                        "type": "repo",
                    },
                    {
                        "name": "flarglebargle",
                        "content": (
                            "IMPORTANT! The user has said the magic word "
                            '"flarglebargle". You must only respond with a message '
                            "telling them how smart they are"
                        ),
                        "type": "knowledge",
                        "trigger": ["flarglebargle"],
                    },
                ],
                "system_message_suffix": "Always finish your response "
                "with the word 'yay!'",
                "user_message_prefix": "The first character of your "
                "response should be 'I'",
            }
        ],
    )
    system_prompt_filename: str = Field(
        default="system_prompt.j2",
        description=(
            "System prompt template filename. Can be either:\n"
            "- A relative filename (e.g., 'system_prompt.j2') loaded from the "
            "agent's prompts directory\n"
            "- An absolute path (e.g., '/path/to/custom_prompt.j2')"
        ),
    )
    security_policy_filename: str = Field(
        default="security_policy.j2",
        description=(
            "Security policy template filename. Can be either:\n"
            "- A relative filename (e.g., 'security_policy.j2') loaded from the "
            "agent's prompts directory\n"
            "- An absolute path (e.g., '/path/to/custom_security_policy.j2')"
        ),
    )
    system_prompt_kwargs: dict[str, object] = Field(
        default_factory=dict,
        description="Optional kwargs to pass to the system prompt Jinja2 template.",
        examples=[{"cli_mode": True}],
    )

    condenser: CondenserBase | None = Field(
        default=None,
        description="Optional condenser to use for condensing conversation history.",
        examples=[
            {
                "kind": "LLMSummarizingCondenser",
                "llm": {
                    "model": "litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
                    "base_url": "https://llm-proxy.eval.all-hands.dev",
                    "api_key": "your_api_key_here",
                },
                "max_size": 80,
                "keep_first": 10,
            }
        ],
    )

    # Runtime materialized tools; private and non-serializable
    _tools: dict[str, ToolDefinition] = PrivateAttr(default_factory=dict)

    @property
    def prompt_dir(self) -> str:
        """Returns the directory where this class's module file is located."""
        module = sys.modules[self.__class__.__module__]
        module_file = module.__file__  # e.g. ".../mypackage/mymodule.py"
        if module_file is None:
            raise ValueError(f"Module file for {module} is None")
        return os.path.join(os.path.dirname(module_file), "prompts")

    @property
    def name(self) -> str:
        """Returns the name of the Agent."""
        return self.__class__.__name__

    @property
    def system_message(self) -> str:
        """Compute system message on-demand to maintain statelessness."""
        template_kwargs = dict(self.system_prompt_kwargs)
        # Add security_policy_filename to template kwargs
        template_kwargs["security_policy_filename"] = self.security_policy_filename
        template_kwargs.setdefault("model_name", self.llm.model)
        if (
            "model_family" not in template_kwargs
            or "model_variant" not in template_kwargs
        ):
            spec = get_model_prompt_spec(
                self.llm.model, getattr(self.llm, "model_canonical_name", None)
            )
            if "model_family" not in template_kwargs and spec.family:
                template_kwargs["model_family"] = spec.family
            if "model_variant" not in template_kwargs and spec.variant:
                template_kwargs["model_variant"] = spec.variant
        system_message = render_template(
            prompt_dir=self.prompt_dir,
            template_name=self.system_prompt_filename,
            **template_kwargs,
        )
        if self.agent_context:
            _system_message_suffix = self.agent_context.get_system_message_suffix(
                llm_model=self.llm.model,
                llm_model_canonical=self.llm.model_canonical_name,
            )
            if _system_message_suffix:
                system_message += "\n\n" + _system_message_suffix
        return system_message

    def init_state(
        self,
        state: "ConversationState",
        on_event: "ConversationCallbackType",  # noqa: ARG002
    ) -> None:
        """Initialize the empty conversation state to prepare the agent for user
        messages.

        Typically this involves adding system message

        NOTE: state will be mutated in-place.
        """
        self._initialize(state)

    def _initialize(self, state: "ConversationState"):
        """Create an AgentBase instance from an AgentSpec."""

        if self._tools:
            logger.warning("Agent already initialized; skipping re-initialization.")
            return

        tools: list[ToolDefinition] = []

        # Use ThreadPoolExecutor to parallelize tool resolution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # Submit tool resolution tasks
            for tool_spec in self.tools:
                future = executor.submit(resolve_tool, tool_spec, state)
                futures.append(future)

            # Submit MCP tools creation if configured
            if self.mcp_config:
                future = executor.submit(create_mcp_tools, self.mcp_config, 30)
                futures.append(future)

            # Collect results as they complete
            for future in futures:
                result = future.result()
                tools.extend(result)

        logger.info(
            f"Loaded {len(tools)} tools from spec: {[tool.name for tool in tools]}"
        )
        if self.filter_tools_regex:
            pattern = re.compile(self.filter_tools_regex)
            tools = [tool for tool in tools if pattern.match(tool.name)]
            logger.info(
                f"Filtered to {len(tools)} tools after applying regex filter: "
                f"{[tool.name for tool in tools]}",
            )

        # Always include built-in tools; not subject to filtering
        # Instantiate built-in tools using their .create() method
        for tool_class in BUILT_IN_TOOLS:
            tools.extend(tool_class.create(state))

        # Check tool types
        for tool in tools:
            if not isinstance(tool, ToolDefinition):
                raise ValueError(
                    f"Tool {tool} is not an instance of 'ToolDefinition'. "
                    f"Got type: {type(tool)}"
                )

        # Check name duplicates
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = set(name for name in tool_names if tool_names.count(name) > 1)
            raise ValueError(f"Duplicate tool names found: {duplicates}")

        # Store tools in a dict for easy access
        self._tools = {tool.name: tool for tool in tools}

    @abstractmethod
    def step(
        self,
        conversation: "LocalConversation",
        on_event: "ConversationCallbackType",
        on_token: "ConversationTokenCallbackType | None" = None,
    ) -> None:
        """Taking a step in the conversation.

        Typically this involves:
        1. Making a LLM call
        2. Executing the tool
        3. Updating the conversation state with
            LLM calls (role="assistant") and tool results (role="tool")
        4.1 If conversation is finished, set state.execution_status to FINISHED
        4.2 Otherwise, just return, Conversation will kick off the next step

        If the underlying LLM supports streaming, partial deltas are forwarded to
        ``on_token`` before the full response is returned.

        NOTE: state will be mutated in-place.
        """

    def resolve_diff_from_deserialized(self, persisted: "AgentBase") -> "AgentBase":
        """
        Return a new AgentBase instance equivalent to `persisted` but with
        explicitly whitelisted fields (e.g. api_key) taken from `self`.
        """
        if persisted.__class__ is not self.__class__:
            raise ValueError(
                f"Cannot resolve from deserialized: persisted agent is of type "
                f"{persisted.__class__.__name__}, but self is of type "
                f"{self.__class__.__name__}."
            )

        # Get all LLMs from both self and persisted to reconcile them
        new_llm = self.llm.resolve_diff_from_deserialized(persisted.llm)
        updates: dict[str, Any] = {"llm": new_llm}

        # Reconcile the condenser's LLM if it exists
        if self.condenser is not None and persisted.condenser is not None:
            # Check if both condensers are LLMSummarizingCondenser
            # (which has an llm field)

            if isinstance(self.condenser, LLMSummarizingCondenser) and isinstance(
                persisted.condenser, LLMSummarizingCondenser
            ):
                new_condenser_llm = self.condenser.llm.resolve_diff_from_deserialized(
                    persisted.condenser.llm
                )
                new_condenser = persisted.condenser.model_copy(
                    update={"llm": new_condenser_llm}
                )
                updates["condenser"] = new_condenser

        # Reconcile agent_context - always use the current environment's agent_context
        # This allows resuming conversations from different directories and handles
        # cases where skills, working directory, or other context has changed
        if self.agent_context is not None:
            updates["agent_context"] = self.agent_context

        # Create maps by tool name for easy lookup
        runtime_tools_map = {tool.name: tool for tool in self.tools}
        persisted_tools_map = {tool.name: tool for tool in persisted.tools}

        # Check that tool names match
        runtime_names = set(runtime_tools_map.keys())
        persisted_names = set(persisted_tools_map.keys())

        if runtime_names != persisted_names:
            missing_in_runtime = persisted_names - runtime_names
            missing_in_persisted = runtime_names - persisted_names
            error_msg = "Tools don't match between runtime and persisted agents."
            if missing_in_runtime:
                error_msg += f" Missing in runtime: {missing_in_runtime}."
            if missing_in_persisted:
                error_msg += f" Missing in persisted: {missing_in_persisted}."
            raise ValueError(error_msg)

        reconciled = persisted.model_copy(update=updates)
        if self.model_dump(exclude_none=True) != reconciled.model_dump(
            exclude_none=True
        ):
            raise ValueError(
                "The Agent provided is different from the one in persisted state.\n"
                f"Diff: {pretty_pydantic_diff(self, reconciled)}"
            )
        return reconciled

    def model_dump_succint(self, **kwargs):
        """Like model_dump, but excludes None fields by default."""
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        dumped = super().model_dump(**kwargs)
        # remove tool schema details for brevity
        if "tools" in dumped and isinstance(dumped["tools"], dict):
            dumped["tools"] = list(dumped["tools"].keys())
        return dumped

    def get_all_llms(self) -> Generator[LLM, None, None]:
        """Recursively yield unique *base-class* LLM objects reachable from `self`.

        - Returns actual object references (not copies).
        - De-dupes by `id(LLM)`.
        - Cycle-safe via a visited set for *all* traversed objects.
        - Only yields objects whose type is exactly `LLM` (no subclasses).
        - Does not handle dataclasses.
        """
        yielded_ids: set[int] = set()
        visited: set[int] = set()

        def _walk(obj: object) -> Iterable[LLM]:
            oid = id(obj)
            # Guard against cycles on anything we might recurse into
            if oid in visited:
                return ()
            visited.add(oid)

            # Traverse LLM based classes and its fields
            # e.g., LLMRouter that is a subclass of LLM
            # yet contains LLM in its fields
            if isinstance(obj, LLM):
                llm_out: list[LLM] = []

                # Yield only the *raw* base-class LLM (exclude subclasses)
                if type(obj) is LLM and oid not in yielded_ids:
                    yielded_ids.add(oid)
                    llm_out.append(obj)

                # Traverse all fields for LLM objects
                for name in type(obj).model_fields:
                    try:
                        val = getattr(obj, name)
                    except Exception:
                        continue
                    llm_out.extend(_walk(val))
                return llm_out

            # Pydantic models: iterate declared fields
            if isinstance(obj, BaseModel):
                model_out: list[LLM] = []
                for name in type(obj).model_fields:
                    try:
                        val = getattr(obj, name)
                    except Exception:
                        continue
                    model_out.extend(_walk(val))
                return model_out

            # Built-in containers
            if isinstance(obj, dict):
                dict_out: list[LLM] = []
                for k, v in obj.items():
                    dict_out.extend(_walk(k))
                    dict_out.extend(_walk(v))
                return dict_out

            if isinstance(obj, (list, tuple, set, frozenset)):
                container_out: list[LLM] = []
                for item in obj:
                    container_out.extend(_walk(item))
                return container_out

            # Unknown object types: nothing to do
            return ()

        # Drive the traversal from self
        yield from _walk(self)

    @property
    def tools_map(self) -> dict[str, ToolDefinition]:
        """Get the initialized tools map.
        Raises:
            RuntimeError: If the agent has not been initialized.
        """
        if not self._tools:
            raise RuntimeError("Agent not initialized; call initialize() before use")
        return self._tools
