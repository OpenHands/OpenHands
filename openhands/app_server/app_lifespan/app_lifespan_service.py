from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import FastAPI

from openhands.sdk.utils.models import DiscriminatedUnionMixin


class AppLifespanService(DiscriminatedUnionMixin, ABC):
    def lifespan(self, api: FastAPI):
        """Return lifespan wrapper."""
        return self

    @abstractmethod
    async def __aenter__(self):
        """Open lifespan."""

    @abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Close lifespan."""
