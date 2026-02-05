# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar(
    'T', int, float
)  # Type for the value (int for iterations, float for budget)


@dataclass
class ControlFlag(Generic[T]):
    """Base class for control flags that manage limits and state transitions."""

    limit_increase_amount: T
    current_value: T
    max_value: T
    headless_mode: bool = False
    _hit_limit: bool = False

    def reached_limit(self) -> bool:
        """Check if the limit has been reached.

        Returns:
            bool: True if the limit has been reached, False otherwise.
        """
        raise NotImplementedError

    def increase_limit(self, headless_mode: bool) -> None:
        """Expand the limit when needed."""
        raise NotImplementedError

    def step(self):
        """Determine the next state based on the current state and mode.

        Returns:
            ControlFlagState: The next state.
        """
        raise NotImplementedError


@dataclass
class IterationControlFlag(ControlFlag[int]):
    """Control flag for managing iteration limits."""

    def reached_limit(self) -> bool:
        """Check if the iteration limit has been reached."""
        self._hit_limit = self.current_value >= self.max_value
        return self._hit_limit

    def increase_limit(self, headless_mode: bool) -> None:
        """Expand the iteration limit by adding the initial value."""
        if not headless_mode and self._hit_limit:
            self.max_value += self.limit_increase_amount
            self._hit_limit = False

    def step(self):
        if self.reached_limit():
            raise RuntimeError(
                f'Agent reached maximum iteration. '
                f'Current iteration: {self.current_value}, max iteration: {self.max_value}'
            )

        # Increment the current value
        self.current_value += 1


@dataclass
class BudgetControlFlag(ControlFlag[float]):
    """Control flag for managing budget limits."""

    def reached_limit(self) -> bool:
        """Check if the budget limit has been reached."""
        self._hit_limit = self.current_value >= self.max_value
        return self._hit_limit

    def increase_limit(self, headless_mode) -> None:
        """Expand the budget limit by adding the initial value to the current value."""
        if self._hit_limit:
            self.max_value = self.current_value + self.limit_increase_amount
            self._hit_limit = False

    def step(self):
        """Check if we've reached the limit and update state accordingly.

        Note: Unlike IterationControlFlag, this doesn't increment the value
        as the budget is updated externally.
        """
        if self.reached_limit():
            current_str = f'{self.current_value:.2f}'
            max_str = f'{self.max_value:.2f}'
            raise RuntimeError(
                f'Agent reached maximum budget for conversation.'
                f'Current budget: {current_str}, max budget: {max_str}'
            )
