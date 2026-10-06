# Inline thinking leaks into the chat bubble (tool-call actions)

Evidence for the Canvas bug where an agent's inline reasoning
(`<think>...</think>` / `reasoning_content`) that belongs to a **tool call**
is rendered verbatim inside the visible chat bubble instead of being hidden
behind the collapsible thinking section.

## How this was produced

Stack: `npm run test:e2e:mock-llm` (the repo's own mock-LLM harness), which
runs the production `bin/agent-canvas.mjs` stack (static frontend + real
agent-server + ingress) against a scripted OpenAI-compatible mock LLM. No real
LLM credentials were used.

The mock LLM was asked to return an assistant tool call whose content is

```
<think>The user asked me to run a command and then reply. ...</think>
Running the command now.
```

plus a following plain text turn (`MOCK_LLM_E2E_REPLY_OK`). The agent-server
records the tool-call text as the `ActionEvent.thought`
(`thought[0].text == "<think>...</think>\nRunning the command now."`).

## What the screenshots show

`leak-page.png` / `leak-full.png` are captured after the conversation settles.

The reasoning text that belongs to the tool call is rendered as a normal
paragraph inside the agent bubble (`<p class="m-0 leading-6">`), fully visible,
instead of being collapsed behind "Thinking". The same content is also absent
from any collapsible thinking region.

## Confirming DOM probe

```json
{"tag":"P","testid":null,"className":"m-0 leading-6",
 "text":"The user asked me to run a command and then reply. I should first check which shell is available, then run the command. ",
 "width":736,"height":144,"visible":true,"display":"block","visibility":"visible"}
```

## Backend events (agent-server 1.49.6)

```
ActionEvent source=agent tool=terminal
  thought=[{"type":"text","text":"<think>...reasoning...</think>\nRunning the command now."}]
ObservationEvent source=environment tool=terminal
MessageEvent source=agent content=["MOCK_LLM_E2E_REPLY_OK"]
```

The `ActionEvent.thought` carries the inline-think text. Canvas's action
renderer (`getActionContent` -> `action.message.trim()`) does not run
`splitInlineThink` on it, so the whole thought — tags and all — reaches the
bubble.
