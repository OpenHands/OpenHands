import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    Self,
    TypeVar,
)

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from openai.types.responses import FunctionToolParam
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema

from openhands.sdk.security import risk
from openhands.sdk.tool.schema import Action, Observation, Schema
from openhands.sdk.utils.models import (
    DiscriminatedUnionMixin,
    get_known_concrete_subclasses,
    kind_of,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


ActionT = TypeVar("ActionT", bound=Action)
ObservationT = TypeVar("ObservationT", bound=Observation)
_action_types_with_risk: dict[type, type] = {}


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case.

    Examples:
        TerminalTool -> bash_tool
        FileEditorTool -> file_editor_tool
        XMLHttpRequest -> xml_http_request
    """
    # Insert underscore before uppercase letters (except the first one)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters that follow lowercase letters
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class ToolAnnotations(BaseModel):
    """Annotations to provide hints about the tool's behavior.

    Based on Model Context Protocol (MCP) spec:
    https://github.com/modelcontextprotocol/modelcontextprotocol/blob/caf3424488b10b4a7b1f8cb634244a450a1f4400/schema/2025-06-18/schema.ts#L838
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        # We need to define the title here to avoid conflict with MCP's ToolAnnotations
        # when both are included in the same JSON schema for openapi.json
        title="openhands.sdk.tool.tool.ToolAnnotations",
    )

    title: str | None = Field(
        default=None, description="A human-readable title for the tool."
    )
    readOnlyHint: bool = Field(
        default=False,
        description="If true, the tool does not modify its environment. Default: false",
    )
    destructiveHint: bool = Field(
        default=True,
        description="If true, the tool may perform destructive updates to its environment. If false, the tool performs only additive updates. (This property is meaningful only when `readOnlyHint == false`) Default: true",  # noqa: E501
    )
    idempotentHint: bool = Field(
        default=False,
        description="If true, calling the tool repeatedly with the same arguments will have no additional effect on the its environment. (This property is meaningful only when `readOnlyHint == false`) Default: false",  # noqa: E501
    )
    openWorldHint: bool = Field(
        default=True,
        description="If true, this tool may interact with an 'open world' of external entities. If false, the tool's domain of interaction is closed. For example, the world of a web search tool is open, whereas that of a memory tool is not. Default: true",  # noqa: E501
    )


class ToolExecutor[ActionT, ObservationT](ABC):
    """Executor function type for a Tool."""

    @abstractmethod
    def __call__(
        self, action: ActionT, conversation: "LocalConversation | None" = None
    ) -> ObservationT:
        """Execute the tool with the given action and return an observation.

        Args:
            action: The action to execute, containing the parameters and context
                   needed for the tool operation.
            conversation: The conversation context for the tool execution.
                         Note: This is typed as LocalConversation (not
                         BaseConversation) because all tool executions happen
                         within a LocalConversation context. Even when tools are
                         invoked via RemoteConversation, the remote agent server
                         creates a LocalConversation instance to handle the actual
                         tool execution. See https://github.com/OpenHands/agent-sdk/pull/925
                         for more details.

        Returns:
            An observation containing the results of the tool execution.
        """

    def close(self) -> None:
        """Close the executor and clean up resources.

        Default implementation does nothing. Subclasses should override
        this method to perform cleanup (e.g., closing connections,
        terminating processes, etc.).
        """
        pass


class ExecutableTool(Protocol):
    """Protocol for tools that are guaranteed to have a non-None executor.

    This eliminates the need for runtime None checks and type narrowing
    when working with tools that are known to be executable.
    """

    name: str
    executor: ToolExecutor[Any, Any]  # Non-optional executor

    def __call__(
        self, action: Action, conversation: "LocalConversation | None" = None
    ) -> Observation:
        """Execute the tool with the given action."""
        ...


class ToolDefinition[ActionT, ObservationT](DiscriminatedUnionMixin, ABC):
    """Base class for all tool implementations.

    This class serves as a base for the discriminated union of all tool types.
    All tools must inherit from this class and implement the .create() method for
    proper initialization with executors and parameters.

    Features:
    - Normalize input/output schemas (class or dict) into both model+schema.
    - Validate inputs before execute.
    - Coerce outputs only if an output model is defined; else return vanilla JSON.
    - Export MCP tool description.

    Examples:
        Simple tool with no parameters:
            class FinishTool(ToolDefinition[FinishAction, FinishObservation]):
                @classmethod
                def create(cls, conv_state=None, **params):
                    return [cls(name="finish", ..., executor=FinishExecutor())]

        Complex tool with initialization parameters:
            class TerminalTool(ToolDefinition[TerminalAction,
                TerminalObservation]):
                @classmethod
                def create(cls, conv_state, **params):
                    executor = TerminalExecutor(
                        working_dir=conv_state.workspace.working_dir,
                        **params,
                    )
                    return [cls(name="terminal", ..., executor=executor)]
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, arbitrary_types_allowed=True
    )

    # Automatic tool naming - set by __init_subclass__
    name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        """Automatically set name from class name when subclass is created."""
        super().__init_subclass__(**kwargs)
        # Only set automatically if not explicitly defined in the current class
        if "name" not in cls.__dict__:
            cls.name = _camel_to_snake(cls.__name__).removesuffix("_tool")

    description: str
    action_type: type[Action] = Field(repr=False)
    observation_type: type[Observation] | None = Field(default=None, repr=False)

    annotations: ToolAnnotations | None = None
    meta: dict[str, Any] | None = None

    # runtime-only; always hidden on dumps
    executor: SkipJsonSchema[ToolExecutor | None] = Field(
        default=None, repr=False, exclude=True
    )

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Sequence[Self]:
        """Create a sequence of Tool instances.

        This method must be implemented by all subclasses to provide custom
        initialization logic, typically initializing the executor with parameters
        from conv_state and other optional parameters.

        Args:
            *args: Variable positional arguments (typically conv_state as first arg).
            **kwargs: Optional parameters for tool initialization.

        Returns:
            A sequence of Tool instances. Even single tools are returned as a sequence
            to provide a consistent interface and eliminate union return types.
        """
        raise NotImplementedError("ToolDefinition subclasses must implement .create()")

    @computed_field(return_type=str, alias="title")
    @property
    def title(self) -> str:
        if self.annotations and self.annotations.title:
            return self.annotations.title
        return self.name

    @field_serializer("action_type")
    def _ser_action_type(self, t: type[Action]) -> str:
        # serialize as a plain kind string
        return kind_of(t)

    @field_serializer("observation_type")
    def _ser_observation_type(self, t: type[Observation] | None) -> str | None:
        return None if t is None else kind_of(t)

    @field_validator("action_type", mode="before")
    @classmethod
    def _val_action_type(cls, v):
        if isinstance(v, str):
            return Action.resolve_kind(v)
        assert isinstance(v, type) and issubclass(v, Action), (
            f"action_type must be a subclass of Action, but got {type(v)}"
        )
        return v

    @field_validator("observation_type", mode="before")
    @classmethod
    def _val_observation_type(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = Observation.resolve_kind(v)
        assert isinstance(v, type) and issubclass(v, Observation), (
            f"observation_type must be a subclass of Observation, but got {type(v)}"
        )
        return v

    def set_executor(self, executor: ToolExecutor) -> Self:
        """Create a new Tool instance with the given executor."""
        return self.model_copy(update={"executor": executor})

    def as_executable(self) -> ExecutableTool:
        """Return this tool as an ExecutableTool, ensuring it has an executor.

        This method eliminates the need for runtime None checks by guaranteeing
        that the returned tool has a non-None executor.

        Returns:
            This tool instance, typed as ExecutableTool.

        Raises:
            NotImplementedError: If the tool has no executor.
        """
        if self.executor is None:
            raise NotImplementedError(f"Tool '{self.name}' has no executor")
        return self  # type: ignore[return-value]

    def action_from_arguments(self, arguments: dict[str, Any]) -> Action:
        """Create an action from parsed arguments.

        This method can be overridden by subclasses to provide custom logic
        for creating actions from arguments (e.g., for MCP tools).

        Args:
            arguments: The parsed arguments from the tool call.

        Returns:
            The action instance created from the arguments.
        """
        return self.action_type.model_validate(arguments)

    def __call__(
        self, action: ActionT, conversation: "LocalConversation | None" = None
    ) -> Observation:
        """Validate input, execute, and coerce output.

        We always return some Observation subclass, but not always the
        generic ObservationT.
        """
        if self.executor is None:
            raise NotImplementedError(f"Tool '{self.name}' has no executor")

        # Execute
        result = self.executor(action, conversation)

        # Coerce output only if we declared a model; else wrap in base Observation
        if self.observation_type:
            if isinstance(result, self.observation_type):
                return result
            return self.observation_type.model_validate(result)
        else:
            # When no output schema is defined, wrap the result in Observation
            if isinstance(result, Observation):
                return result
            elif isinstance(result, BaseModel):
                return Observation.model_validate(result.model_dump())
            elif isinstance(result, dict):
                return Observation.model_validate(result)
            raise TypeError(
                "Output must be dict or BaseModel when no output schema is defined"
            )

    def to_mcp_tool(
        self,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a Tool to an MCP tool definition.

        Allow overriding input/output schemas (usually by subclasses).

        Args:
            input_schema: Optionally override the input schema.
            output_schema: Optionally override the output schema.
        """
        out = {
            "name": self.name,
            "description": self.description,
            "inputSchema": input_schema or self.action_type.to_mcp_schema(),
        }
        if self.annotations:
            out["annotations"] = self.annotations
        if self.meta is not None:
            out["_meta"] = self.meta

        derived_output = (
            output_schema
            if output_schema is not None
            else (
                self.observation_type.to_mcp_schema() if self.observation_type else None
            )
        )
        if derived_output is not None:
            out["outputSchema"] = derived_output
        return out

    def _get_tool_schema(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> dict[str, Any]:
        action_type = action_type or self.action_type
        action_type_with_risk = create_action_type_with_risk(action_type)

        add_security_risk_prediction = add_security_risk_prediction and (
            self.annotations is None or (not self.annotations.readOnlyHint)
        )
        schema = (
            action_type_with_risk.to_mcp_schema()
            if add_security_risk_prediction
            else action_type.to_mcp_schema()
        )
        return schema

    def to_openai_tool(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> ChatCompletionToolParam:
        """Convert a Tool to an OpenAI tool.

        Args:
            add_security_risk_prediction: Whether to add a `security_risk` field
                to the action schema for LLM to predict. This is useful for
                tools that may have safety risks, so the LLM can reason about
                the risk level before calling the tool.
            action_type: Optionally override the action_type to use for the schema.
                This is useful for MCPTool to use a dynamically created action type
                based on the tool's input schema.
        """
        return ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name=self.name,
                description=self.description,
                parameters=self._get_tool_schema(
                    add_security_risk_prediction, action_type
                ),
            ),
        )

    def to_responses_tool(
        self,
        add_security_risk_prediction: bool = False,
        action_type: type[Schema] | None = None,
    ) -> FunctionToolParam:
        """Convert a Tool to a Responses API function tool (LiteLLM typed).

        For Responses API, function tools expect top-level keys:
        { "type": "function", "name": ..., "description": ..., "parameters": ... }
        """

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._get_tool_schema(
                add_security_risk_prediction, action_type
            ),
            "strict": False,
        }

    @classmethod
    def resolve_kind(cls, kind: str) -> type:
        """Resolve a kind string to its corresponding tool class.

        Args:
            kind: The name of the tool class to resolve

        Returns:
            The tool class corresponding to the kind

        Raises:
            ValueError: If the kind is unknown
        """
        for subclass in get_known_concrete_subclasses(cls):
            if subclass.__name__ == kind:
                return subclass

        # Get all possible kinds for the error message
        possible_kinds = [
            subclass.__name__ for subclass in get_known_concrete_subclasses(cls)
        ]
        possible_kinds_str = (
            ", ".join(sorted(possible_kinds)) if possible_kinds else "none"
        )

        error_msg = (
            f"Unexpected kind '{kind}' for {cls.__name__}. "
            f"Expected one of: {possible_kinds_str}. "
            f"If you receive this error when trying to wrap a DiscriminatedUnion "
            f"instance inside another pydantic model, you may need to use "
            f"OpenHandsModel instead of BaseModel to make sure that an invalid "
            f"schema has not been cached."
        )
        raise ValueError(error_msg)


def create_action_type_with_risk(action_type: type[Schema]) -> type[Schema]:
    action_type_with_risk = _action_types_with_risk.get(action_type)
    if action_type_with_risk:
        return action_type_with_risk

    action_type_with_risk = type(
        f"{action_type.__name__}WithRisk",
        (action_type,),
        {
            "security_risk": Field(
                # We do NOT add default value to make it an required field
                # default=risk.SecurityRisk.UNKNOWN
                description="The LLM's assessment of the safety risk of this action.",
            ),
            "__annotations__": {"security_risk": risk.SecurityRisk},
        },
    )
    _action_types_with_risk[action_type] = action_type_with_risk
    return action_type_with_risk
