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
4. Replace the placeholder LiteLLM key in `config.toml`.
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

## Notes
- The container needs `/var/run/docker.sock` because OpenHands local GUI launches sandbox/runtime containers.
- This fork keeps upstream OpenHands functionality intact and adds Persian UI support plus QADR-specific deployment packaging.
