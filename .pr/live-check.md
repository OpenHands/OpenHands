# End-to-end check: the three System prompt modes in real Canvas conversations

Full local stack (`npm run dev`: ingress, automation and Vite), with the agent-server built from software-agent-sdk#5438 (`OH_AGENT_SERVER_LOCAL_PATH`), state isolated in a scratch dir, and the recording mock LLM (`tests/e2e/mock-llm/scripts/mock-llm-server.py`). The mock returns a fixed "Mock LLM reply." regardless of the prompt. The proof is what each conversation stored and what the LLM was sent. `npm run build` passes on this branch.

Playwright drove the real UI:
1. Authored `explorer` with **Custom prompt** (`01-editor-custom-prompt.png`).
2. Authored `helper` with **Default + your instructions** (`02-editor-instructions.png`).
3. Created a named `plain` profile with the OpenHands default.
4. Reopened both authored profiles.
5. Launched each from the home chat and opened **Agent Tools & Metadata → System Message** (`03`–`05`). `04` is cropped to the dynamic block so local user skills stay out of the screenshot.

```
stored explorer: system_prompt==PROMPT true, suffix null
stored helper:   system_prompt null, suffix==INSTRUCTIONS true
reopen explorer: text ok true, hint "Replaces the built-in OpenHands instruct…"
reopen helper:   text ok true, hint "Your instructions are added after the bu…"
[explorer] LLM static block == PROMPT: true  | built-in <ROLE> sent: false | instructions sent: false | modal: PROMPT true,  <SOUL> false, instructions false
[helper]   LLM static block == PROMPT: false | built-in <ROLE> sent: true  | instructions sent: true  | modal: PROMPT false, <SOUL> true,  instructions true
[plain]    LLM static block == PROMPT: false | built-in <ROLE> sent: true  | instructions sent: false | modal: PROMPT false, <SOUL> true,  instructions false
page errors: none
```
