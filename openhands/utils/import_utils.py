import importlib
from functools import lru_cache
from typing import TypeVar

T = TypeVar('T')


def import_from(qual_name: str):
    """Import a value from its fully qualified name.

    This function is a utility to dynamically import any Python value (class, function, variable)
    from its fully qualified name. For example, 'openhands.server.user_auth.UserAuth' would
    import the UserAuth class from the openhands.server.user_auth module.

    Args:
        qual_name: A fully qualified name in the format 'module.submodule.name'
                  e.g. 'openhands.server.user_auth.UserAuth'

    Returns:
        The imported value (class, function, or variable)

    Example:
        >>> UserAuth = import_from('openhands.server.user_auth.UserAuth')
        >>> auth = UserAuth()
    """
    parts = qual_name.split('.')
    module_name = '.'.join(parts[:-1])
    module = importlib.import_module(module_name)
    result = getattr(module, parts[-1])
    return result


@lru_cache()
def _get_impl(cls: type[T], impl_name: str | None) -> type[T]:
    if impl_name is None:
        return cls
    impl_class = import_from(impl_name)
    assert cls == impl_class or issubclass(impl_class, cls)
    return impl_class


def get_impl(cls: type[T], impl_name: str | None) -> type[T]:
    """Import and validate a named implementation of a base class.

    This function is an extensibility mechanism in OpenHands that allows runtime substitution
    of implementations. It enables applications to customize behavior by providing their own
    implementations of OpenHands base classes.

    The function ensures type safety by validating that the imported class is either the same as
    or a subclass of the specified base class.

    Args:
        cls: The base class that defines the interface
        impl_name: Fully qualified name of the implementation class, or None to use the base class
                  e.g. 'openhands.server.conversation_manager.StandaloneConversationManager'

    Returns:
        The implementation class, which is guaranteed to be a subclass of cls

    Example:
        >>> # Get default implementation
        >>> ConversationManager = get_impl(ConversationManager, None)
        >>> # Get custom implementation
        >>> CustomManager = get_impl(ConversationManager, 'myapp.CustomConversationManager')

    Common Use Cases:
        - Server components (ConversationManager, UserAuth, etc.)
        - Storage implementations (ConversationStore, SettingsStore, etc.)
        - Service integrations (GitHub, GitLab, Bitbucket, Azure DevOps services)

    The implementation is cached to avoid repeated imports of the same class.
    """
    return _get_impl(cls, impl_name)  # type: ignore
