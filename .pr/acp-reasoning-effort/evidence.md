# Live evidence: ACP reasoning effort in the running Canvas

Canvas `dev:minimal` against a local agent-server built from
software-agent-sdk#6107 (`bf28aa90e`), with `@openhands/typescript-client`
packed from that branch (`npm install --no-save`) so the registry lists
`effort_levels`. Pinned adapters (claude-agent-acp 0.63.0, codex-acp 2.2.2),
real Claude and ChatGPT subscription logins. A Playwright script drove the
composer picker; effort was read where the provider records it.

## Claude Code (`opus[1m]`)

```
home pick High           -> saved acp_model: opus[1m]/high
turn 1 chip              -> Opus (1M context) · High
live switch to Low       -> chip: Opus (1M context) · Low; llm_model opus[1m]/low
Anthropic /v1/messages   -> claude-opus-5 output_config={"effort": "high"} (2 msgs)
                            claude-opus-5 output_config={"effort": "low"}  (4 msgs, same session)
```

## Codex (`gpt-6.1-sol`)

```
home pick Max            -> saved acp_model: gpt-6.1-sol/max
turn 1 chip              -> 6.1 Sol · Max
live switch to Ultra     -> chip: 6.1 Sol · Ultra; llm_model gpt-6.1-sol/ultra
rollout turn_context     -> (gpt-6.1-sol, max), (gpt-6.1-sol, ultra)
```

## Settings → Agent

The profile saved as `gpt-6.1-sol/max` opens as Model "6.1 Sol" + Reasoning
Effort "Max", and the form stays clean (Save disabled).
