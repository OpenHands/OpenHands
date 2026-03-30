# QADR Deployment

This fork is prepared for a Persian-first deployment on `hands.gantor.ir`.

## Scope
- Public GUI for OpenHands on QADR
- Persian available in the UI language selector
- RTL-aware document language handling for Persian and Arabic
- Reverse proxy through the shared QADR Caddy ingress

## Files
- [`compose.qadr.yaml`](/C:/Users/never/Documents/CodeX/gantor-openhands/compose.qadr.yaml)
- [`.env.qadr.example`](/C:/Users/never/Documents/CodeX/gantor-openhands/.env.qadr.example)
- [`config.template.toml`](/C:/Users/never/Documents/CodeX/gantor-openhands/config.template.toml)
- [`config.qadr.example.toml`](/C:/Users/never/Documents/CodeX/gantor-openhands/config.qadr.example.toml)

## Recommended host paths
- state: `/srv/data/openhands-state`
- workspace: `/srv/data/openhands-workspace`

## Minimal rollout on QADR
1. Create the host directories.
2. Copy `.env.qadr.example` to `.env.qadr` and adjust values if needed.
3. Create `config.toml` inside the state directory from `config.qadr.example.toml`.
4. Replace the placeholder LiteLLM key in `config.toml` with a dedicated service key.
5. Start the stack:

```bash
docker compose --env-file .env.qadr -f compose.qadr.yaml up -d --build
```

## Reverse proxy
The source-of-truth ingress block is maintained in the QADR `freegpt` repository:
- [`Caddyfile`](/C:/Users/never/Documents/CodeX/freegpt/stacks/ingress-core/Caddyfile)

Expected site:
- `hands.gantor.ir -> qadr-openhands:3000`

## DNS
Add:

```dns
hands    IN A    5.235.208.128
```

to the authoritative `gantor.ir` zone used by QADR.

## Gantor Platform Integration
OpenHands is part of the Gantor platform alongside:
- **WorldMonitor** at `monitor.gantor.ir` — geopolitical intelligence dashboard
- **FreeGPT** at `freegpt.ir` / `chat.freegpt.ir` — free AI chat

WorldMonitor links to OpenHands and FreeGPT from its header, footer, and mobile menu.

## Notes
- The container needs `/var/run/docker.sock` because OpenHands local GUI launches sandbox/runtime containers.
- The QADR compose file mounts `config.toml` into `/app/config.toml` so the runtime can deterministically load the live service configuration.
- The recommended LLM path is the internal FreeGPT/LiteLLM API on `http://qadr-ai-gateway-litellm:4000/v1`, backed by a dedicated OpenHands service key rather than the LiteLLM master key.
- This fork keeps upstream OpenHands functionality intact and adds Persian UI support plus QADR-specific deployment packaging.
