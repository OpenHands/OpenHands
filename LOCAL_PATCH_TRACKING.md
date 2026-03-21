# Local Patch Tracking (andyelka-creator fork)

This fork is used as a temporary integration carrier for reproducible fixes and documentation before upstream merge.

## Why this exists

Our deployment runs OpenHands in a remote VM with an external OpenAI-compatible gateway (LiteLLM), mixed local and premium model lanes, and stricter operational constraints than a default single-host setup.

## Branch policy

- `main`: mirror of upstream `OpenHands/OpenHands` (no long-lived local drift).
- `patch/*`: short-lived branches, one issue/fix per branch.
- every patch branch must have:
  - reproducible steps;
  - before/after behavior;
  - rollback note.

## Current upstream references

- https://github.com/OpenHands/OpenHands/issues/13475
- https://github.com/OpenHands/OpenHands/issues/13476
- https://github.com/OpenHands/OpenHands/issues/13477
- https://github.com/OpenHands/OpenHands/issues/13478
- https://github.com/OpenHands/OpenHands/issues/13479

## Contribution rule

Local patches are temporary. The target state is upstream merge or local patch removal.
