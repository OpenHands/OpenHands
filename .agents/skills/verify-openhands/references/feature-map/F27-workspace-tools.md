# F27 — Workspace tools: terminal, browser, planner, tasks and usage

DRAFT in progress (being re-driven live). Sub-features below are proven unless marked otherwise.

Source: `src/components/features/terminal/`, `src/routes/browser-tab.tsx`, `src/routes/planner-tab.tsx`, `src/routes/task-list-tab.tsx`, `src/routes/usage-tab.tsx`, `src/components/features/conversation/usage-panel/`, `src/services/canvas-ui.ts`.

## Sub-features

- `F27.terminal-empty`: proven.
- `F27.terminal-output`: proven.
- `F27.terminal-history-reload`: FAIL (#17566 reproduces).
- `F27.terminal-live-append`: proven.
- `F27.browser-empty`: proven.
- `F27.planner-empty`: proven.
- `F27.planner-plan`: proven.
- `F27.planner-build`: proven.
- `F27.tasklist`: proven.
- `F27.usage-metrics`: proven.
- `F27.usage-compact`: proven.
- `F27.usage-provider-balance`: hidden-card case proven.
- `F27.composer-context-meter`: proven.
- `F27.canvas-ui-open-tab`: proven.

## How to get to it (user POV)

- Conversation header panel button (`right-panel-toggle`), then the drawer tabs.

## Driving it with control-openhands

Preconditions:

- Baseline state; `control-openhands llm preset deepseek`.

## Gotchas

- Terminal history is lost after reload (#17566).
