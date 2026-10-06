# Inline reasoning leaks into the chat bubble for tool-call turns

Evidence for github.com/OpenHands/OpenHands#18074.

An agent's inline reasoning that belongs to a **tool call** used to render
verbatim inside the visible chat bubble instead of behind the collapsible
thinking section. The same words then read as duplicated once the final reply
arrived, which is how the bug was reported.

## How this was produced

Stack: `npm run test:e2e:mock-llm` — the repo's own mock-LLM harness, which
runs the production `bin/agent-canvas.mjs` stack (static frontend + real
agent-server + ingress) against a scripted OpenAI-compatible mock LLM. No real
LLM credentials are used; the mock only makes the trigger deterministic.

The scripted turn is an assistant tool call whose content is

```
<think>Let me check the working directory before running it.</think>
Running the command now.
```

followed by a plain text reply. The agent-server stores the tool-call content as
the `ActionEvent.thought`:

```
ActionEvent source=agent tool=terminal
  thought=[{"type":"text","text":"<think>...</think>\nRunning the command now."}]
```

## Before / after

| Artifact | Build | Result |
|---|---|---|
| `before-fix.png` | `thought-event-message.tsx` unpatched | reasoning text visible in the agent bubble; no thinking section |
| `after-fix.png` | `thought-event-message.tsx` patched | reasoning inside the collapsible thinking section; bubble shows only "Running the command now." |

Both runs use the same spec (`tests/e2e/mock-llm/regressions/mock-llm-inline-think-leak.spec.ts`)
and the same scripted trajectory, so the only variable is the fix.

Before the fix the spec fails at `expect(collapsible-thinking).toBeVisible()`,
and Playwright's ARIA snapshot shows the leaked paragraph in the bubble:

```
- paragraph: Let me check the working directory before running it. Running the command now.
```

After the fix the same spec passes, with
`bubbleHasReasoning=false bubbleHasRawTag=false thinkingBlocks=1`.

`leak-page.png` / `leak-full.png` are the original wider captures from the same
harness that accompanied the bug report, including the DOM probe below.

## Confirming DOM probe (pre-fix)

```json
{"tag":"P","testid":null,"className":"m-0 leading-6",
 "text":"The user asked me to run a command and then reply. ...",
 "width":736,"height":144,"visible":true,"display":"block","visibility":"visible"}
```

## Root cause

`ThoughtEventMessage` passed the raw `ActionEvent.thought` straight to
`ChatMessage`. `splitInlineThink` already existed and was wired into the message
paths (`user-assistant-event-message.tsx`, and the streaming-delta branch of
`event-message.tsx`), but not into the action-thought path.
