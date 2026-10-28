# Task List: Reply / quote a message in the chat

Source: `tasks/plan.md` (implements the approved spec
`docs/specs/reply-message.md`, intent in `docs/intent/reply-message.md`).

## Phase 1: Foundation

- [x] Task 1: Envelope format module (`src/utils/message-quote.ts` + round-trip test suite)
- [x] Task 2: Armed-quote store + `pendingReplyQuote` persistence/sanitization
- [x] Task 3: i18n keys (6 × 16 locales) + `make-i18n`

## Checkpoint: Foundation

- [ ] Format round-trip tests green
- [ ] Store hydration tests green
- [x] i18n completeness passes; `npm run lint` clean
- [ ] Review with human before proceeding

## Phase 2: Core feature

- [ ] Task 4: Composer quote card (slot, dismiss, Escape; jump stub)
- [ ] Task 5: Compose at submit + retry replays envelope
- [ ] Task 6: Reply action on bubbles + surface gating

## Checkpoint: Core feature

- [ ] End-to-end: hover → Reply → card → send → envelope in fixture
- [ ] Retry keeps envelope; disable-path audit clean
- [ ] Review with human before proceeding

## Phase 3: Rendering + jump

- [ ] Task 7: Bubble extraction & truncation isolation (sent + pending)
- [ ] Task 8: Jump to message (`data-event-id`, load-until-found, 10-page guard)
- [ ] Task 9: E2E regression scenario + full command gate

## Checkpoint: Complete

- [ ] All 10 spec success criteria met
- [ ] `npm run lint`, `npm test`, `npm run build`, `npm run build:lib`,
      `npm run check-translation-completeness` all green
- [ ] Ready for review/PR
