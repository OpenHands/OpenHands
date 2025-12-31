from typing import Self, overload

from openhands.sdk.logger import get_logger
from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.local import LocalWorkspace
from openhands.sdk.workspace.remote.base import RemoteWorkspace


logger = get_logger(__name__)


class Workspace:
    """Factory entrypoint that returns a LocalWorkspace or RemoteWorkspace.

    Usage:
        - Workspace(working_dir=...) -> LocalWorkspace
        - Workspace(working_dir=..., host="http://...") -> RemoteWorkspace
    """

    @overload
    def __new__(
        cls: type[Self],
        *,
        working_dir: str = "workspace/project",
    ) -> LocalWorkspace: ...

    @overload
    def __new__(
        cls: type[Self],
        *,
        host: str,
        working_dir: str = "workspace/project",
        api_key: str | None = None,
    ) -> RemoteWorkspace: ...

    def __new__(
        cls: type[Self],
        *,
        host: str | None = None,
        working_dir: str = "workspace/project",
        api_key: str | None = None,
    ) -> BaseWorkspace:
        if host:
            return RemoteWorkspace(
                working_dir=working_dir,
                host=host,
                api_key=api_key,
            )
        return LocalWorkspace(working_dir=working_dir)
