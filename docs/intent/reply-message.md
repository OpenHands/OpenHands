# Intent: Reply to a message in a conversation

Status: confirmed by the user (interview, 2026-10-28). This is the source of
truth for the feature; specs and plans derive from it.

## Outcome

A user can reply to any message in the chat — an earlier message of their own
or an agent reply — and the replied-to message is quoted **in full** as a
structured quote block inside the outgoing user message. Long messages are
truncated automatically as a token safeguard, with an explicit marker showing
the quote was cut.

## User

Users of the standard chat tab who need a follow-up to one specific message
without copying its text again.

## Why now

The model reads the quote as specific context rather than relying on the
ordinal flow of the conversation — the same result as reply in Perplexity and
Muse AI. The quote-in-prompt approach was chosen because it delivers that
result today, on both Local and Cloud backends, without a protocol change.

## Success

- A Reply action is available in the existing per-bubble hover `actions` slot
  in `ChatMessage`, for both user and agent messages.
- Invoking Reply replaces the composer contents with the quote and places the
  caret at the end. An unsent draft is overwritten (decided in the interview).
- While composing, a quote card is visible above the composer.
- After a reload the quote still renders, because it is rebuilt from the
  message body rather than from transient UI state.
- The quote card is clickable and jumps to the original message.

## Constraint

Frontend-only in this Canvas repository. No changes to `SendMessageRequest`
or `MessageEvent`, no agent-server changes, no raise of
`compatibility.minimumAgentServer`, and no changes to the fork or
planning-agent paths.

## Out of scope

- Reply-to-selection: the whole message is quoted, not a selected range.
- Protocol-level `reply_to_event_id` (the rejected option B, owned by
  `software-agent-sdk`).
- Threads or stacking multiple quotes in one message.
- Reply from surfaces other than the main chat tab: the planning
  sub-conversation, BTW messages, and other docked composers.
- Any change in `software-agent-sdk`.
