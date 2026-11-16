# Live evidence: ACP reasoning effort as `acp_effort`

Setup:
- Canvas `dev:minimal` against a local agent-server built from
  software-agent-sdk#6556.
- `@openhands/typescript-client` packed from that branch and installed with
  `npm install --no-save`.
- Pinned adapters (`claude-agent-acp@0.63.0`, `codex-acp@2.2.2`), real Claude
  and ChatGPT subscription logins, and a PATH without global adapters.

A Playwright script did four things through the composer picker:
1. Picks an effort on the home page.
2. Sends a prompt.
3. Switches the effort live.
4. Switches the model live.

Effort was read where each provider records it: the Anthropic `/v1/messages`
body (`output_config.effort`, captured by a local proxy on `ANTHROPIC_BASE_URL`)
and the Codex rollout `turn_context`.

## Claude Code (`opus[1m]`)

```
home pick Low            -> profile saved {acp_model: opus[1m], acp_effort: low}
turn 1                   chip "Opus (1M context) · Low"; session effort low
live switch High         chip "… · High"; session effort high
live model switch Haiku  chip "Haiku", no effort row; session effort null
/v1/messages             (claude-opus-5, low, 2 msgs), (claude-opus-5, high, 4 msgs),
                         (claude-haiku-4-5-20251001, none, 5 msgs)
```

## Codex (`gpt-6.1-sol`)

```
home pick Max                  -> profile saved {acp_model: gpt-6.1-sol, acp_effort: max}
turn 1                         chip "6.1 Sol · Max"
live switch Ultra              chip "6.1 Sol · Ultra"
live model switch 5.6 Luna     chip "5.6 Luna · Medium"; effort row Low..Max (no Ultra)
rollout turn_context           (gpt-6.1-sol, max), (gpt-6.1-sol, ultra), (gpt-5.6-luna, medium)
```

## Settings → Agent

The `effort-field-codex` profile opens with Reasoning Effort "Max", listing the
levels gpt-6.1-sol offers (Low … Ultra) plus Agent default. Save stays
disabled because the form is clean.
