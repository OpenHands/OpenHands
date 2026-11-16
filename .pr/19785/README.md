# Per-provider upload limits in the LLM settings tab

Baseline: `035cffa` (`main`).

The real production Canvas build ran against the local Agent Server 1.60.1 on
`http://127.0.0.1:18000`. The bundle was served by the repository's own
`scripts/static-server.mjs`, so the runtime-config injection path is the one
the Docker and npm launchers use.

## What the screenshot shows

`llm-settings-provider-limit.png` is the **Settings → LLM → Edit profile LLM**
screen with the provider set to **Anthropic**. Under the provider and model
fields the note reads:

> Anthropic aceita imagens de até 10MB. Outros anexos são limitados a 3MB.

Two facts are visible at once:

- **10MB for images** — the deployment override
  `AGENT_CANVAS_PROVIDER_UPLOAD_LIMITS='{"anthropic":10}'`, passed to the
  static server as `--provider-upload-limits '{"anthropic":10}'`, is what the
  UI reports for Anthropic. The same server injected
  `window.__AGENT_CANVAS_PROVIDER_UPLOAD_LIMITS__="{\"anthropic\":10}"` into
  the served `index.html`.
- **3MB for every other attachment** — a PDF, a video or an archive is
  uploaded to the workspace and read from there by the agent, so the provider
  never receives its bytes and the provider's image limit says nothing about
  it. Those attachments keep the generic 3MB ceiling.

Selecting a provider with no published limit (the screenshot before this one
shows `minimax`) renders the fallback wording instead, so the number is never
mistaken for a documented provider limit:

> Não há limite de imagem publicado conhecido para minimax, então as imagens
> são limitadas a 3MB.

## Runtime injection, verified against the served HTML

```console
$ node scripts/static-server.mjs --port 3113 --dir build --host 127.0.0.1 \
    --provider-upload-limits '{"anthropic":10}'
$ curl -s http://127.0.0.1:3113/ | grep -o '__AGENT_CANVAS_PROVIDER_UPLOAD_LIMITS__[^<]*'
__AGENT_CANVAS_PROVIDER_UPLOAD_LIMITS__="{\"anthropic\":10}";}());
```

Without the flag the global is absent, so a deployment that configures
nothing keeps the compiled-in per-provider limits.

## Automated verification

- `npm run lint` — 0 errors (132 pre-existing warnings, none in the changed
  files).
- `npx tsc --noEmit -p tsconfig.json` — clean, apart from two pre-existing
  `./+types/*` route-module errors that also reproduce on an unmodified
  `main`.
- `npm run build:app` — succeeds.
- `npx vitest run` — 10 757 passed, 3 failed. All three failures reproduce on
  a clean `main` checkout in this environment:
  `.agents/skills/verify-openhands/scripts/control-openhands.test.mjs`
  (flaky under parallel load; passes in isolation) and
  `__tests__/scripts/docker-entrypoint-preflight.test.ts` (the unreadable-file
  case does not fail as root).
