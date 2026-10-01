# End-to-end check: a profile's system prompt in a real Canvas conversation

Full local stack (`npm run dev`: ingress, automation and Vite), with the agent-server built from software-agent-sdk#5438 (`OH_AGENT_SERVER_LOCAL_PATH`). State was isolated in a scratch dir. The LLM is the recording mock (`tests/e2e/mock-llm/scripts/mock-llm-server.py`), which returns a fixed "Mock LLM reply." regardless of the prompt. The proof is what the conversation stored and what the LLM was sent. `npm run build` also passes on this branch.

Playwright drove the real UI:
1. Settings → Agent → Add agent profile `explorer`, System prompt → Custom prompt, typed the prompt, Save (`01-editor-custom-prompt.png`).
2. Opened the `default` profile (`02-editor-default-profile.png`).
3. Created a named `plain` profile with no custom prompt, as a paired control.
4. For each profile: activated it, sent a message from the home chat, waited for the reply (`03-explorer-conversation.png`), then opened **Agent Tools & Metadata → System Message** (`03-explorer-system-prompt.png`, `04-plain-system-prompt.png`).

```
save blocked while custom prompt empty: true
stored system_prompt == PROMPT: true
[explorer] LLM block 1 == PROMPT: true  | built-in <ROLE> sent: false | blocks: 2 | modal shows PROMPT: true  | modal shows built-in <SOUL>: false
[plain]    LLM block 1 == PROMPT: false | built-in <ROLE> sent: true  | blocks: 2 | modal shows PROMPT: false | modal shows built-in <SOUL>: true
page errors: none
```
