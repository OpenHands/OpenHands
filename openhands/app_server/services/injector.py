import contextlib
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generic, TypeAlias, TypeVar

from fastapi import Request
from starlette.datastructures import State

T = TypeVar('T')
InjectorState: TypeAlias = State


class Injector(Generic[T], ABC):
    """Object designed to facilitate dependency injection"""

    @abstractmethod
    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[T, None]:
        """Inject an object. The state object may be used to store variables for
        reuse by other injectors, as injection operations may be nested."""
        yield None  # type: ignore

    @contextlib.asynccontextmanager
    async def context(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[T, None]:
        """Context function suitable for use in async with clauses"""
        async for result in self.inject(state, request):
            yield result

    async def depends(self, request: Request) -> AsyncGenerator[T, None]:
        """Depends function suitable for use with FastAPI dependency injection."""
        async for result in self.inject(request.state, request):
            yield result
