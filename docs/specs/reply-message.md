# Spec: Reply / quote a message in the chat

Status: APPROVED — open questions answered by the user (2026-10-28), see
Resolved Decisions. Consumes the confirmed intent in
`docs/intent/reply-message.md`; that document owns scope, this document owns
the design.

## Objective

Let a user follow up on one specific message — their own or the agent's — by
replying to it. The replied-to message is quoted in full inside the outgoing
user message, so the model reads the exact context rather than guessing from
conversation order. Everything stays frontend-only: no `SendMessageRequest` /
`MessageEvent` change, no agent-server change, no
`compatibility.minimumAgentServer` bump.

Success looks like: hovering a user or agent bubble shows a Reply action;
clicking it arms the composer with a quote card; sending delivers
`envelope + follow-up` as one normal user message; the pending bubble, the
echoed bubble, and every later reload render the quote card the same way;
clicking a quote card scrolls to the original message.

## ASSUMPTIONS (correct me now or I proceed with these)

1. **Envelope-in-text, card-in-UI.** The quote travels inside the message text
   as a self-describing envelope (format below). The composer shows a *card*,
   never raw markup; the envelope is composed at submit time and re-parsed for
   display. This is the only mechanism that survives reload without wire
   changes — it is what "quote-in-prompt" means concretely.
2. **One quote per message, no nesting.** Quoting a message that is itself a
   reply quotes only that message's visible follow-up text (inner envelope
   stripped). Keeps the format depth-1 and the parser trivial.
3. **Text only.** Images in the source message are not quoted. An image-only
   message (parses to `""`) gets no Reply action — same rule the branch
   feature uses at `user-assistant-event-message.tsx:84-86`.
4. **Truncation cap 2000 chars** on the quoted body, flagged as
   `truncated="true"` in the envelope and rendered as a localized label.
5. **Reply replaces the draft** (Q3a): clicking Reply clears unsent composer
   text, arms the quote, caret at the end of the now-empty composer.
6. **Mid-compose reload survives**: the armed quote persists in
   `ConversationState` localStorage next to `draftMessage`, with a sanitizer
   rule, so composing continues after reload.
7. **Jump load-until-found, bounded.** The jump helper keeps calling
   `useLoadOlderEvents.loadOlder()` while pages remain, so the target is
   found regardless of depth — the user's call ("mekanisme terbaik demi
   hasil terbaik") outranks a small page budget; the existing
   `loading-older-events` spinner covers the wait. A hard guard still caps
   runaway loops if history is huge (10 pages), then the localized "message
   not in loaded history" hint appears instead of scrolling blindly.
8. **Surfaces**: Reply exists only in the main chat tab of a standard
   (non-plan) conversation. Hidden on planning-agent bubbles
   (`isFromPlanningAgent`), in plan mode, on BTW messages, on pending
   "sending" bubbles before they exist as events, and on non-message events.
   It works identically for Local and Cloud backends and for OpenHands and ACP
   agents — the quote is plain text, so nothing backend-specific exists.

## Wire format (the one contract this feature owns)

A pure-frontend text envelope, start-anchored in the user message text.
Tokens are program identifiers (repo Rule 2: named constants, never inline
literals) and are locale-independent:

```
<reply-to id="<source event id>" source="user|agent" truncated="false">
<quoted body, verbatim, inner envelopes stripped>
</reply-to>
<user's follow-up text>
```

- Only the text part of `content` is touched; the existing
  `sendMessageWithAttachments` files-prompt stays appended after it.
- Parser rules: anchored at position 0 (same convention as
  `splitInlineThink`), at most one envelope per message, a follow-up that
  already begins with the opening tag is left untouched (idempotent compose),
  malformed/absent envelope degrades to plain rendering — never a crash.
- `id` is the source `MessageEvent.id` (ULID/UUID string). It exists for both
  sources; matching to a rendered bubble is a new capability (see Jump).

## Design

### New module: `src/utils/message-quote.ts`

Pure, React-free, fully unit-testable owner of the format:

```ts
export const REPLY_OPEN_TAG = "<reply-to";
// …

export interface ReplyQuote {
  sourceEventId: string;
  source: "user" | "agent";
  body: string;        // quoted text actually stored (post-truncation)
  truncated: boolean;
}

export function composeReplyText(quote: ReplyQuote | null, followUp: string): string;
export function splitReplyFromText(text: string): { quote: ReplyQuote | null; body: string };
export function quoteBodyForEvent(event: MessageEvent, fullText: string): ReplyQuote | null;
```

- `quoteBodyForEvent` derives the quote from the source bubble's *displayed*
  text (the same string the user sees — agent messages already went through
  `splitInlineThink`), truncates at the cap, strips inner envelopes, and
  returns `null` for image-only/empty sources.
- No exporter or transcript tool changes: the envelope is ordinary text and
  must survive `src/utils/transcript-export` untouched.

### State owner: `src/stores/reply-quote-store.ts`

One zustand store, one writer per action, conversation-scoped — mirrors
`optimistic-user-message-store` conventions:

- `setReplyQuote(conversationId, quote)` — writer of the Reply action.
- `clearReplyQuote(conversationId)` — writer of dismiss + submit.
- Selector filtered by active conversation, like `PendingUserMessages` does,
  so switching conversations never carries an armed quote over.
- Hydration: `ConversationState` in `src/utils/conversation-local-storage.ts`
  gains `pendingReplyQuote` (validated in `sanitizeStoredState`: object shape
  check, known `source` values, bounded `body` length — corrupt entries are
  dropped, per the file's existing sanitizer pattern). Cleared on submit and
  on dismiss; skipped for task/temporary conversation ids, like
  `draftMessage`.

### Composer: card above the field

- `QuoteCard` (`src/components/features/chat/quote-card.tsx`) renders in
  `ChatInputContainer` beside the existing `<UploadedFiles />` row
  (`chat-input-container.tsx:83`) — the established chip-row precedent.
- Card contents: localized source label ("You"/"Agent"), quoted preview
  (markdown-rendered, own clamp), truncation label when flagged, a dismiss
  button, and click-to-jump to the source message.
- Submit path (`use-chat-submission.ts`) composes
  `composeReplyText(quote, typedText)` in the same transaction that already
  reads + clears the composer, then clears the quote store. Attachments flow
  through unchanged, so the files prompt stays after the follow-up text.
- Retry of a failed pending message replays the stored `content` verbatim
  (`pending-user-messages.tsx:77-95`), so a retried reply keeps its envelope.

### Bubbles: extract, don't truncate the quote

- In `user-assistant-event-message.tsx` (and the pending bubble in
  `pending-user-messages.tsx`), run `splitReplyFromText` on the parsed message
  text: `quote` renders as a `QuoteCard` inside/above the bubble; only
  `body` feeds the existing truncation logic.
- This is required, not cosmetic: `UserMessageBody`'s collapse rule
  (`newlineCount >= 5`, `length > 360` at `user-message-body.tsx:8-10,58-61`)
  would otherwise hide the user's own follow-up behind a 5-line clamp the
  moment the quote exceeds it. Without extraction the feature visually
  swallows every reply.
- The fork/"edit message" path keeps today's behavior with the full original
  text (envelope included as text) — the branch feature is out of scope.

### Jump to message (new capability)

Today nothing anchors events in the DOM — no `data-event-id` anywhere, and
the scroll context only offers bottom-scrolling
(`use-scroll-to-bottom.ts:63-76`). So:

- `ChatMessage` gains an optional `eventId` prop rendered as
  `data-event-id` — a product anchor for jumping, not a test hook.
- `src/hooks/chat/use-jump-to-event.ts`: find
  `[data-event-id="<id>"]` inside the `ScrollProvider` container
  (`chat-interface.tsx:601-602`), `scrollIntoView({ block: "center",
  behavior: "smooth" })` ( precedent:
  `activity-log-section.tsx:83-86`), plus a brief highlight pulse. If absent,
  call `useLoadOlderEvents.loadOlder()` repeatedly while `hasMore` holds —
  load-until-found, bounded by a 10-page guard against runaway loops — then
  surface the localized "not in loaded history" hint. Escape also dismisses
  the armed quote card (keyboard parity with the mouse-only X).

### i18n

New keys, all 16 locales per repo rule (edit `src/i18n/translation.json`, run
`npm run make-i18n`, then `scripts/check-translation-completeness.cjs`):

- `CHAT_INTERFACE$REPLY` — hover action tooltip/aria-label
- `CHAT_INTERFACE$REPLY_YOU` / `CHAT_INTERFACE$REPLY_AGENT` — card source label
- `CHAT_INTERFACE$REPLY_TRUNCATED` — truncation label
- `CHAT_INTERFACE$REPLY_DISMISS` — card dismiss aria-label
- `CHAT_INTERFACE$REPLY_NOT_LOADED` — jump failure hint

Envelope tokens are NOT i18n keys (Rule 2 constants). Existing prefix
`CHAT_INTERFACE$` reused; no new prefix.

## Commands

```
npm ci                                   # node >= 24, npm 11.12.1 pinned
npm run lint                             # eslint 9 flat config
npm test                                 # vitest (runs make-i18n first)
npm run build                            # standalone app build
npm run build:lib                        # library build + d.ts
npm run make-i18n                        # after any translation.json edit
npm run check-translation-completeness   # staged-file check too
```

## Project Structure

```
src/utils/message-quote.ts                       # format: compose/split/derive (pure)
src/stores/reply-quote-store.ts                  # armed-quote state, one writer per action
src/components/features/chat/quote-card.tsx      # shared card (composer + bubbles)
src/hooks/chat/use-jump-to-event.ts              # scroll-to-event + bounded page loads
src/components/features/chat/chat-message.tsx    # + eventId prop → data-event-id
src/components/conversation-events/chat/
  event-message-components/user-assistant-event-message.tsx  # + Reply action + extraction
src/components/features/chat/pending-user-messages.tsx      # extraction for pending bubbles
src/components/features/chat/components/chat-input-container.tsx  # card slot
src/hooks/chat/use-chat-submission.ts            # compose at submit + clear quote
src/utils/conversation-local-storage.ts          # + pendingReplyQuote + sanitizer
src/i18n/translation.json                        # + 5 keys × 16 locales
__tests__/utils/message-quote.test.ts            # new: format round-trip suite
tests/e2e/mock-llm/regressions/                  # + reply scenario in the UI-regressions suite
```

## Code Style

Follow the surrounding chat code exactly — named-constant envelope tokens,
`I18nKey` for every visible string, semantic color tokens only
(`shadcn/no-raw-colors`), no inline styles, `cn()` for conditional classes.

```tsx
// The Reply action rides the existing hover-actions slot — no new visual system:
const actions = [
  ...existing,
  ...(canReply
    ? [{
        icon: <ReplyIcon width={15} height={15} aria-hidden />,
        onClick: handleReply,
        tooltip: t(I18nKey.CHAT_INTERFACE$REPLY),
      }]
    : []),
];
```

## Testing Strategy

Vitest + jsdom + Testing Library, matching existing placement and the repo
testing baseline (Arrange/Act/Assert, behavior over presentation, mock the
service not the hook consumer).

- **New `__tests__/utils/message-quote.test.ts`** — the core: compose/split
  round-trip, start-anchored parsing, idempotent compose, truncation flag,
  inner-envelope stripping, malformed-envelope passthrough, image-only source
  → null.
- **Extend `user-assistant-event-message.test.tsx`** — Reply action gating
  (user/agent source, planning bubble, plan mode); click arms store +
  composer; stored envelope renders as card after "reload" (fresh mount from
  event); fork path untouched.
- **Extend `chat-message.test.tsx`** — quote card inside bubble; only
  follow-up text drives truncation; `data-event-id` present; card click
  invokes the jump (scrollIntoView + center).
- **Extend `pending-user-messages.test.tsx`** — pending bubble shows the same
  card; retry replays envelope.
- **Extend the mock-llm UI regressions e2e** — one scenario: reply to an
  agent message, send, reload, assert card, click card, assert scroll target.

## Boundaries

- **Always:** run the full command list before done; new visible strings in
  all 16 locales via translation.json (never hand-edit generated files);
  envelope tokens as named constants; keep quote rendering identical across
  pending/sent/reloaded states; deterministic tests.
- **Ask first:** new npm dependency (none expected); any new public export
  from `src/lib/index.ts`; widening the Reply action to more surfaces;
  changing the envelope format after spec approval.
- **Never:** touch `software-agent-sdk`, the TS wire types in
  `agent-server-conversation-service.types.ts`, `compatibility.minimumAgentServer`,
  the planning sub-conversation/BTW/docked-composer flows, or the
  verify-openhands skill files; delete a failing test to pass.

## Success Criteria

1. Reply action appears on hover for user and agent message bubbles in the
   standard main chat tab, keyboard-accessible with localized aria-label.
2. Clicking Reply replaces the draft (Q3a), arms the quote card above the
   composer, caret at the end; dismiss (X) clears both; switching
   conversation or submitting clears it.
3. Sent message content is exactly `envelope + follow-up` (plus the existing
   files prompt when attachments exist) — verifiable in MSW/websocket
   fixtures.
4. Pending "sending" bubble, echoed bubble, and reload-from-history render
   the identical quote card; quoted text never counts toward the follow-up's
   truncation measurement.
5. A quote of 6+ lines leaves the user's follow-up visible (no clamped-only
   bubble).
6. Truncated quotes flag `truncated="true"` and render the localized
   truncation label; the 2000-char cap holds.
7. Clicking a quote card scrolls the source bubble to center with a visible
   highlight; unrendered targets auto-load older pages until found (10-page
   guard) before the "not in loaded history" hint appears.
8. Reply is absent on planning bubbles, plan mode, BTW, pending pre-event
   bubbles, image-only messages, and all non-message events.
9. `npm run lint`, `npm test`, `npm run build`, `npm run build:lib`,
   `npm run check-translation-completeness` all pass; transcript export still
   works with envelope text present.
10. Escape dismisses the armed quote card.

## Resolved Decisions (review, 2026-10-28)

1. **Truncation cap**: 2000 chars — approved.
2. **Jump**: load-until-found with a 10-page guard, not a 5-page cap —
   chosen for the best result regardless of history depth.
3. **Escape key**: included; dismisses the armed quote card.
