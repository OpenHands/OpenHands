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

## Evidence scope (defensive handling)

This reproduction uses the repo's scripted **mock** LLM (OpenAI-compatible
fixture served by `tests/e2e/mock-llm/scripts/mock-llm-server.py`; the fixture
identifies itself as model `mock-preflight`). No real LLM credentials are
available in the development environment, so the trigger is produced by a
scripted trajectory rather than a live model.

The frontend defect itself is model-independent: once an `ActionEvent.thought`
contains a leading inline reasoning block, Canvas renders it verbatim. The mock
only makes that payload deterministic. We therefore describe the fix as
**defensive handling of inline reasoning in action thoughts** rather than a
reproduction against a specific production model, and the mock-LLM spec is
regression coverage, not live-model evidence. The code path is shared by every
backend/model, so the same input reaches it in production whenever a model emits
reasoning inline.

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

## Follow-up: duplicate thinking sections

Review of the first fix found a second, related defect. `EventMessage` rendered
explicit `reasoning_content` / `thinking_blocks` as its own `CollapsibleThinking`
and then invoked `ThoughtEventMessage`, which rendered the inline block as
another one. An action carrying both therefore produced **two thinking
controls**, and when the two held the same text it appeared twice — visible in
`thought-event-message.tsx` around R38.

`ThoughtEventMessage` is now the single owner: `splitActionNarration()` returns
`{ reasoning, message }` where `reasoning` merges explicit reasoning with a
leading inline block (`mergeReasoning` drops an exact duplicate), and
`EventMessage` renders explicit reasoning only when `ThoughtEventMessage` does
not own it — on the action path, the observation replacement path, and the
hoisted-thought path in `messages.tsx`. `getActionNarration` uses the same split,
so a downloaded transcript no longer writes raw `think` tags or repeated
reasoning.

Verification for this follow-up is deterministic at the component level rather
than a second screenshot, because the mock trajectory cannot emit
`reasoning_content`:

- `event-thought-helpers.test.ts` — `splitActionNarration` merges identical
  explicit + inline reasoning into one fragment and keeps distinct fragments.
- `event-message-think-action.test.tsx` — a `Messages` render of an action with
  `reasoning_content: "Check the directory"` plus an identical inline block
  asserts exactly one `collapsible-thinking` whose text appears once, alongside
  the `Running ls` bubble.
- `transcript-export/index.test.ts` — the exported markdown contains no raw tags
  and the reasoning once.
