<!-- Keep this PR as draft until it is ready for review. -->

HUMAN:

<!-- Human contributors: add a short note about your testing. -->

---

AGENT:

Adds Maritime as the second provider in Add Backend → Providers, next to DigitalOcean. Design doc: `.pr/maritime/design.html` (htmlpreview link to be pinned to the pushed commit SHA). Mock-mode captures of every new screen are in `.pr/maritime/*.png`; live captures against a real Maritime account are still to be added (see How to Test).

## Why

The Providers tab has one provider and says "More providers coming soon." [Maritime](https://maritime.sh) runs each agent in its own Firecracker micro-VM with a persistent disk and a public HTTPS + WebSocket hostname, sleeping idle VMs with their memory kept. That makes an OpenHands agent-server on Maritime an ordinary remote backend: no tunnel, no proxy. This follows the DigitalOcean pattern, as discussed with Robert Brennan.

## Summary

- **Trusted-process bridge** (`scripts/maritime-api.mjs`, `scripts/maritime-bridge.mjs`, `scripts/maritime-credentials.mjs`): the Maritime REST client and credential store live in the Electron main process (and, in the web build, the Canvas server behind `MARITIME_WEB=1`, `scripts/maritime-web-bridge.mjs`); the renderer reaches them only as `window.maritimeBridge`. The Maritime API key never reaches the page. Creating an agent builds the whole `POST /api/agents` body in the trusted process: the published-image-based agent-server image tagged with the agent-server version Canvas pins, `publicWeb`, and generated `OH_SESSION_API_KEYS_0` / `OH_SECRET_KEY` secrets (plus `OH_ALLOW_CORS_ORIGINS_0` for a non-loopback Canvas origin).
- **UI** (`src/components/features/backends/maritime/`, `backend-form-modal.tsx`): a Maritime row in Providers; sign in with an API key; the account's plan and its OpenHands agents (running / sleeping / starting / failed); New agent (name + memory); Open (wakes a sleeping agent, registers its hostname as a `kind: "local"` backend with its session key, lands in its latest conversation); Sleep; Delete and "Set new access key" behind `ConfirmationModal`.
- **Lifecycle and cost**: `useBackendsHealth` treats a non-active Maritime backend like a non-active MARS session (probed once, no polling), because every request wakes the VM. Shared web-bridge guards move from `mars-web-bridge.mjs` into `scripts/web-bridge-guards.mjs` with no behavior change.

## Issue Number

Fixes #<issue>

## How to Test

Unit and component tests:

```bash
npm run lint
npx vitest run __tests__/scripts/maritime-api.test.ts __tests__/scripts/maritime-bridge.test.ts __tests__/scripts/maritime-web-bridge.test.ts __tests__/api/maritime-web-bridge.test.ts __tests__/scripts/dev-static.test.ts __tests__/scripts/mars-web-bridge.test.ts
npx vitest run __tests__/components/backends/add-backend-modal.test.tsx
npm test
npm run build && npm run build:lib
```

Live, against a Maritime account (desktop):

1. Create a Maritime API key at https://maritime.sh/settings/api-keys with the provision, deploy and secrets scopes.
2. `npm run desktop`, then Add Backend → Providers → Maritime, paste the key.
3. New agent → name it, keep 4 GB → Create and open. The first start pulls the image (a few minutes). Canvas lands in a new chat on the agent; configure an LLM in Settings and run a task.
4. Back in Providers → Maritime: Sleep the agent, then Open it again (wakes in about a second, same conversations). Delete it.

Web build: `MARITIME_WEB=1 npm run dev`, same steps in the browser.

## Video/Screenshots

Mock mode (`npm run dev:mock`, fake `window.maritimeBridge`, no Maritime account), see `.pr/maritime/capture-screenshots.mjs`:

- `01-providers.png`: Providers lists Maritime under DigitalOcean.
- `02-sign-in.png`: API key sign-in.
- `03-agents.png`: account bar and agents in three states.
- `04-new-agent.png`: New agent form.
- `05-delete-confirmation.png`: delete confirmation.

Live captures (real Maritime account, real agent-server, real LLM): to be added.

## Design Doc

`.pr/maritime/design.html`
