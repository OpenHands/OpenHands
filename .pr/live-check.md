# End-to-end check: persona, instructions and default in real Canvas conversations

Full local stack (`npm run dev`: ingress, automation and Vite), with the agent-server built from software-agent-sdk#5449 (`OH_AGENT_SERVER_LOCAL_PATH`). It advertises `profile_persona_v1`, and not `profile_system_prompt_v1`. State was isolated in a scratch dir, and the LLM is the recording mock (`tests/e2e/mock-llm/scripts/mock-llm-server.py`). The mock returns a fixed "Mock LLM reply." regardless of the prompt. The proof is what each conversation stored and what the LLM was sent. `npm run build` passes on this branch.

Playwright drove the real UI:
1. Authored `explorer` with **Custom persona** (`01-editor-persona.png`).
2. Authored `helper` with **Default + your instructions** (`02-editor-instructions.png`).
3. Created a named `plain` profile with the OpenHands default.
4. Reopened both authored profiles.
5. Launched each from the home chat and opened **Agent Tools & Metadata → System Message**:
   - `03-persona-system-message.png`: the persona, followed by the kept `<MEMORY>` and `<SECURITY>`.
   - `03-persona-capabilities-kept.png`: the same conversation scrolled to `<SECURITY_RISK_ASSESSMENT>`.
   - `04-instructions-system-message.png`: cropped to the dynamic block.
   - `05-default-system-message.png`.

```
stored explorer: persona==PERSONA true, suffix null, system_prompt undefined
stored helper:   persona null, suffix==INSTRUCTIONS true
reopen explorer: text ok true, hint "Replaces OpenHands' built-in persona and…"
reopen helper:   text ok true, hint "Your instructions are added after the bu…"
[explorer] static starts with persona: true  | kept: true | persona-layer sent: none | instructions sent: false | modal: persona true,  <SOUL> false, <SECURITY_RISK_ASSESSMENT> true
[helper]   static starts with persona: false | kept: true | persona-layer sent: <SOUL>,<ROLE>,<CODE_QUALITY>,<VERSION_CONTROL>,<PULL_REQUESTS> | instructions sent: true  | modal: persona false, <SOUL> true, <SECURITY_RISK_ASSESSMENT> true
[plain]    static starts with persona: false | kept: true | persona-layer sent: <SOUL>,<ROLE>,<CODE_QUALITY>,<VERSION_CONTROL>,<PULL_REQUESTS> | instructions sent: false | modal: persona false, <SOUL> true, <SECURITY_RISK_ASSESSMENT> true
page errors: none
```

`kept` means all of `<MEMORY>`, `<SECURITY>`, `<SECURITY_RISK_ASSESSMENT>`, `<BROWSER_TOOLS>`, `<EXTERNAL_SERVICES>` and `<PROCESS_MANAGEMENT>` reached the LLM.
