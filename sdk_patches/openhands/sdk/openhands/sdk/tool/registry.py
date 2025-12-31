import inspect
from collections.abc import Callable, Sequence
from threading import RLock
from typing import TYPE_CHECKING, Any

from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool
from openhands.sdk.tool.tool import ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

logger = get_logger(__name__)

# A resolver produces ToolDefinition instances for given params.
Resolver = Callable[[dict[str, Any], "ConversationState"], Sequence[ToolDefinition]]
"""A resolver produces ToolDefinition instances for given params.

Args:
    params: Arbitrary parameters passed to the resolver. These are typically
        used to configure the ToolDefinition instances that are created.
    conversation: Optional conversation state to get directories from.
Returns: A sequence of ToolDefinition instances. Most of the time this will be a
    single-item
    sequence, but in some cases a ToolDefinition.create may produce multiple tools
    (e.g., BrowserToolSet).
"""

_LOCK = RLock()
_REG: dict[str, Resolver] = {}
_MODULE_QUALNAMES: dict[str, str] = {}  # Maps tool name to module qualname


def _resolver_from_instance(name: str, tool: ToolDefinition) -> Resolver:
    if tool.executor is None:
        raise ValueError(
            "Unable to register tool: "
            f"ToolDefinition instance '{name}' must have a non-None .executor"
        )

    def _resolve(
        params: dict[str, Any], _conv_state: "ConversationState"
    ) -> Sequence[ToolDefinition]:
        if params:
            raise ValueError(
                f"ToolDefinition '{name}' is a fixed instance; params not supported"
            )
        return [tool]

    return _resolve


def _resolver_from_callable(
    name: str, factory: Callable[..., Sequence[ToolDefinition]]
) -> Resolver:
    def _resolve(
        params: dict[str, Any], conv_state: "ConversationState"
    ) -> Sequence[ToolDefinition]:
        try:
            # Try to call with conv_state parameter first
            created = factory(conv_state=conv_state, **params)
        except TypeError as exc:
            raise TypeError(
                f"Unable to resolve tool '{name}': factory could not be called with "
                f"params {params}."
            ) from exc
        if not isinstance(created, Sequence) or not all(
            isinstance(t, ToolDefinition) for t in created
        ):
            raise TypeError(
                f"Factory '{name}' must return Sequence[ToolDefinition], "
                f"got {type(created)}"
            )
        return created

    return _resolve


def _is_abstract_method(cls: type, name: str) -> bool:
    try:
        attr = inspect.getattr_static(cls, name)
    except AttributeError:
        return False
    # Unwrap classmethod/staticmethod
    if isinstance(attr, (classmethod, staticmethod)):
        attr = attr.__func__
    return getattr(attr, "__isabstractmethod__", False)


def _resolver_from_subclass(_name: str, cls: type[ToolDefinition]) -> Resolver:
    create = getattr(cls, "create", None)

    if create is None or not callable(create) or _is_abstract_method(cls, "create"):
        raise TypeError(
            "Unable to register tool: "
            f"ToolDefinition subclass '{cls.__name__}' must define .create(**params)"
            f" as a concrete classmethod"
        )

    def _resolve(
        params: dict[str, Any], conv_state: "ConversationState"
    ) -> Sequence[ToolDefinition]:
        created = create(conv_state=conv_state, **params)
        if not isinstance(created, Sequence) or not all(
            isinstance(t, ToolDefinition) for t in created
        ):
            raise TypeError(
                f"ToolDefinition subclass '{cls.__name__}' create() must return "
                f"Sequence[ToolDefinition], "
                f"got {type(created)}"
            )
        # Optional sanity: permit tools without executor; they'll fail at .call()
        return created

    return _resolve


def register_tool(
    name: str,
    factory: ToolDefinition
    | type[ToolDefinition]
    | Callable[..., Sequence[ToolDefinition]],
) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ToolDefinition name must be a non-empty string")

    if isinstance(factory, ToolDefinition):
        resolver = _resolver_from_instance(name, factory)
    elif isinstance(factory, type) and issubclass(factory, ToolDefinition):
        resolver = _resolver_from_subclass(name, factory)
    elif callable(factory):
        resolver = _resolver_from_callable(name, factory)
    else:
        raise TypeError(
            "register_tool(...) only accepts: (1) a ToolDefinition instance with "
            ".executor, (2) a ToolDefinition subclass with .create(**params), or "
            "(3) a callable factory returning a Sequence[ToolDefinition]"
        )

    # Track the module qualname for this tool
    module_qualname = None
    if isinstance(factory, type):
        module_qualname = factory.__module__
    elif callable(factory):
        module_qualname = getattr(factory, "__module__", None)
    elif isinstance(factory, ToolDefinition):
        module_qualname = factory.__class__.__module__

    with _LOCK:
        # TODO: throw exception when registering duplicate name tools
        if name in _REG:
            logger.warning(f"Duplicate tool name registerd {name}")
        _REG[name] = resolver
        if module_qualname:
            _MODULE_QUALNAMES[name] = module_qualname


def resolve_tool(
    tool_spec: Tool, conv_state: "ConversationState"
) -> Sequence[ToolDefinition]:
    with _LOCK:
        resolver = _REG.get(tool_spec.name)

    if resolver is None:
        raise KeyError(f"ToolDefinition '{tool_spec.name}' is not registered")

    return resolver(tool_spec.params, conv_state)


def list_registered_tools() -> list[str]:
    with _LOCK:
        return list(_REG.keys())


def get_tool_module_qualnames() -> dict[str, str]:
    """Get a mapping of tool names to their module qualnames.

    Returns:
        A dictionary mapping tool names to module qualnames (e.g.,
        {"glob": "openhands.tools.glob.definition"}).
    """
    with _LOCK:
        return dict(_MODULE_QUALNAMES)
