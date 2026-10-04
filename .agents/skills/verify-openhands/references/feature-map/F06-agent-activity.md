# F06 — Agent activity: messages, events and lifecycle

DRAFT in progress (F06 mapping agent). Sections below are proven so far; the rest is being driven.

Source: `src/components/features/chat/chat-interface.tsx`, `src/components/conversation-events/chat/`, `src/components/features/chat/`, `src/components/features/markdown/`, `src/components/shared/buttons/conversation-confirmation-buttons.tsx`.

## Sub-features

- `F06.empty-state-suggestions`: an empty conversation shows "Let's start building!" with four chips; a chip fills the composer.
- `F06.pending-messages`: a sent message shows "Sending..." until echoed; a failed send shows Failed to send with Retry and Dismiss.
- `F06.error-banner`: losing the Agent Server shows "Unable to connect to server" with Retry, Copy and Close.
- `F06.confirmation-mode`: with confirmation mode on, the agent waits with Cancel / Continue.

## How to get to it (user POV)

- Any `/conversations/<id>` page.

## Driving it with control-openhands

Preconditions:

- Baseline state with `control-openhands llm preset deepseek`.

- **Draft (`F06.empty-state-suggestions`).** Run `control-openhands browser click 'testid=conversation-panel-new-thread-picker'`, then `control-openhands browser click 'testid=launch-no-workspace' --expect-url '/conversations/'`.

## Gotchas

- Draft.
