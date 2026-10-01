# Live check: system prompt authored in the editor reaches the LLM

`npm run dev:minimal` with `OH_AGENT_SERVER_LOCAL_PATH` pointing at software-agent-sdk#5438, and the recording mock LLM (`tests/e2e/mock-llm/scripts/mock-llm-server.py`). Playwright drove the real UI:

1. Settings → Agent → Add agent profile `explorer`, System prompt → Custom prompt, typed the prompt, Save.
2. Reopened it from the row menu.
3. Opened the `default` profile.
4. Activated `explorer`, sent a message from the home chat, and read the agent-step request the mock LLM recorded.

```
save disabled while custom prompt empty: true
stored system_prompt == PROMPT: true
reopened editor shows stored prompt: true
default profile shows hint, no editor: true
launched system message block 1 == PROMPT: true
built-in <ROLE> absent: true
dynamic block kept: 2
page errors: none
```
