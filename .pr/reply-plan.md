# Implementation Plan: Reply / quote a message in the chat

## Overview

Implement the approved spec `docs/specs/reply-message.md`: hovering a user or
agent bubble offers Reply; the composer arms a quote card; sending composes a
start-anchored `<reply-to>` envelope plus the follow-up into the normal user
message; pending/sent/reloaded bubbles re-render the quote card; clicking a
card jumps to the source message (load-until-found, 10-page guard). Strictly
frontend-only — no SDK, wire-type, or compatibility-floor changes.

## Architecture Decisions

- **Envelope-in-text, card-in-UI** — quote travels inline so it survives
  reload with zero wire changes; the composer never shows raw markup.
- **`src/utils/message-quote.ts` is the single format owner** — pure,
  React-free, fully unit-testable; every consumer (store, submit, bubbles,
  pending) imports from it. No parsing logic anywhere else.
- **`reply-quote-store` is the one state owner** — writers: Reply click
  (`setReplyQuote`), dismiss/submit (`clearReplyQuote`). Conversation-scoped
  like `optimistic-user-message-store`.
- **Extraction at render, truncation isolation** — quote cards are pulled out
  of the message body before `UserMessageBody` measures, so a 6+ line quote
  can never swallow the follow-up behind the 5-line clamp.
- **Jump is net-new** — `data-event-id` anchor on `ChatMessage` +
  `use-jump-to-event` driving the existing `ScrollProvider` ref and
  `useLoadOlderEvents`; i18n keys land once, up front.

## Dependency Graph

```
message-quote.ts (pure format) ──┬── reply-quote-store (+ localStorage hydration)
 i18n keys (5 × 16 locales)      │        │
                                 │        ├── QuoteCard in composer (slot, dismiss, Escape)
                                 │        │        └── compose-at-submit + clear (retry replay)
                                 │        └── Reply action on bubbles (gating)
                                 └── bubble extraction → QuoteCard in bubbles
                                          (sent + pending + reload) + truncation isolation
                                                   │
                                                   └── jump: data-event-id + use-jump-to-event
        horizontal share: e2e regression scenario sits on top of 5 & 6
```

## Task List

### Phase 1: Foundation

- [x] **Task 1: Envelope format module** — `src/utils/message-quote.ts` with
  `REPLY_OPEN_TAG`/`REPLY_CLOSE_TAG` constants, `ReplyQuote` type,
  `composeReplyText`, `splitReplyFromText`, `quoteBodyForEvent` (derives from
  displayed text, 2000-char cap, strips inner envelopes, null for empty
  sources). New `__tests__/utils/message-quote.test.ts`: compose/split
  round-trip, start-anchor, idempotent compose, truncation flag, inner
  envelope stripping, malformed passthrough, image-only → null.
  **Verify:** `npx vitest run __tests__/utils/message-quote.test.ts`; `npm run lint`.
  **Depends:** none. **Scope:** S (2 files).

- [x] **Task 2: Armed-quote store + persistence** — `src/stores/
  reply-quote-store.ts` (conversation-scoped, writers only per Architecture);
  `pendingReplyQuote` field in `ConversationState` +
  `sanitizeStoredState` validation (shape/`source` whitelist/bounded length)
  in `src/utils/conversation-local-storage.ts`; hydrate on store init, clear
  on submit/dismiss, skip task-ids like `draftMessage` (extend the existing
  `conversation-local-storage` test file if present).
  **Verify:** focused store + storage tests; `npm run lint`.
  **Depends:** Task 1. **Scope:** M (3-4 files).

- [x] **Task 3: i18n keys** — add the 5 keys from the spec
  (`CHAT_INTERFACE$REPLY`, `$REPLY_YOU`, `$REPLY_AGENT`,
  `$REPLY_TRUNCATED`, `$REPLY_DISMISS`, `$REPLY_NOT_LOADED` — 6 total) in all
  locales; run `npm run make-i18n`; no source edits in this task.
  **Verify:** `npm run make-i18n && npm run check-translation-completeness`.
  **Depends:** none (needed by 4, 5, 7). **Scope:** S (1 file + generated).

### Checkpoint: Foundation

- [ ] Format round-trip tests green; store hydration tests green; i18n
  completeness passes; `npm run lint` clean.
- [ ] Review with human before proceeding.

### Phase 2: Core feature

- [x] **Task 4: Composer quote card** — `QuoteCard` component; render slot in
  `ChatInputContainer` beside `<UploadedFiles />`; dismiss (X) +
  Escape-to-dismiss; click-to-jump stub wired later in Task 7 (button present,
  jump handler optional prop). Localized labels only.
  **Verify:** new `quote-card.test.tsx` + container slot test; `npm run lint`.
  **Depends:** Tasks 2, 3. **Scope:** M (3-4 files).

- [x] **Task 5: Compose at submit + retry** — `use-chat-submission` composes
  `composeReplyText(quote, typedText)` in the same transaction that reads and
  clears the composer, then `clearReplyQuote`; failed-send retry replays
  stored `content` verbatim (envelope included).
  **Verify:** extend submission + `pending-user-messages` tests; `npm run lint`.
  **Depends:** Tasks 1, 2, 4. **Scope:** S (2-3 files).

- [x] **Task 6: Reply action on bubbles** — extend the `actions` array in
  `user-assistant-event-message.tsx`: Reply icon for user + agent sources;
  gating = main chat tab only (not planning bubbles / plan mode / BTW /
  pending pre-event bubbles / image-only); onClick `setReplyQuote` + replace
  draft via existing `setMessageToSend("")`. Extend
  `user-assistant-event-message.test.tsx` for gating + arm behavior.
  **Verify:** focused tests; `npm run lint`.
  **Depends:** Tasks 1, 2, 3. **Scope:** M (2-3 files).

### Checkpoint: Core feature

- [x] End-to-end in standard chat: hover → Reply → card → send → envelope
  lands in MSW/websocket fixture; failed send → retry keeps envelope.
- [x] Disable-path audit: no Reply on planning/plan/BTW/pending/image-only.
- [ ] Review with human before proceeding.

### Phase 3: Rendering + jump

- [x] **Task 7: Bubble extraction & truncation isolation** — apply
  `splitReplyFromText` in `user-assistant-event-message.tsx` and
  `pending-user-messages.tsx`; render QuoteCard above the body; only the
  follow-up feeds `UserMessageBody` truncation; card identical across
  pending/sent/reloaded.
  **Verify:** extend `chat-message.test.tsx` + `pending-user-messages`
  tests; `npm run lint`.
  **Depends:** Tasks 1, 4. **Scope:** M (3-4 files).

- [x] **Task 8: Jump to message** — `ChatMessage` optional `eventId` prop →
  `data-event-id`; `src/hooks/chat/use-jump-to-event.ts` (find anchor in
  ScrollProvider container, `scrollIntoView` center + highlight pulse,
  load-until-found via `useLoadOlderEvents` with 10-page guard, localized
  hint on failure); wire card clicks in composer + bubbles.
  **Verify:** jump tests (found/not-found paths); `npm run lint`.
  **Depends:** Tasks 4, 7. **Scope:** M (4-5 files).

- [x] **Task 9: E2E regression + full gate** — one mock-llm regression
  scenario (reply → send → reload → card → click → scroll target); run the
  full command list from the spec: `npm run lint`, `npm test`, `npm run
  build`, `npm run build:lib`, `npm run check-translation-completeness`.
  **Verify:** all five commands green.
  **Depends:** Tasks 7, 8. **Scope:** M (2-3 files).

### Checkpoint: Complete

- [x] All 10 spec success criteria demonstrably met.
- [x] Full command list green; ready for review/PR.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Envelope parsing edge cases break display | High | Pure module + exhaustive round-trip/malformed tests before any UI consumes it (Task 1 gates everything) |
| `UserMessageBody` collapse swallows follow-up | High | Extraction isolation is its own task (7) with a dedicated 6+ line test |
| Jump target deep in unloaded history | Med | Load-until-found + 10-page guard + localized hint (spec-resolved behavior) |
| Draft persistence interplay (prefill/restore) | Med | Reuse existing `setMessageToSend`/draft paths only; no second writer; store tests cover conversation switches |
| i18n completeness blocks commit | Low | Keys land in one dedicated task (3) with the check run in-task |
| Transcript/export contains raw envelope | Low | Accepted per spec: envelope is ordinary text (Boundary "Never" forbids export changes) |

## Open Questions

None — spec decisions resolved and approved (see spec "Resolved Decisions").

## Parallelization

Tasks 1 and 3 are independent (pure module vs translation JSON) and could run
concurrently; Task 2 depends only on 1. Tasks 4/5/6 fan out after 2+3. Tasks
7 and 8 share `QuoteCard`/`ChatMessage` touch points — sequential.
