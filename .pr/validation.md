# Validation of OpenHands #17372

Implementation head: `d4f8b9392b63df74ecf1b5434be83f70388866d1`.
SDK counterpart: https://github.com/OpenHands/software-agent-sdk/pull/5452.
Documentation: https://github.com/OpenHands/docs/pull/866.

## Local verification

Windows native Agent Server, Python 3.13.5, Node 24.11.1. MSW disabled for live
browser verification. The SDK PR's built TypeScript `dist` was copied into
`node_modules/@openhands/typescript-client/dist` for local verification. No
unreleased dependency pin is committed. A clean install of this draft cannot
build until the matching client release is published and exactly pinned.

Commands and observed results:

- `npm run lint`: passed typecheck, ESLint and Prettier (376 pre-existing warnings).
- `npm run build`: application build passed.
- `npm run build:lib`: library build passed.
- `npm run check-translation-completeness`: all 15 locales passed.
- `npx vitest run __tests__/components/settings/codex-auth-card.test.tsx __tests__/components/settings/acp-credentials-section.test.tsx __tests__/components/onboarding/setup-acp-secrets-step.test.tsx __tests__/hooks/query/use-acp-auth-status.test.tsx`: 46 tests passed.
- The same SDK counterpart passed 81 backend/cross tests and 348 client tests.

## Real browser evidence

Started the SDK branch's full `create_app(Config(...))` on loopback `:18731`
with persistent storage, stable encryption key, isolated CODEX_HOME and a
session API key. Optional VSCode and tool preload were disabled; the optional
browser-use tool was unavailable in this Windows environment.

Started the built Canvas with:

```sh
node scripts/static-server.mjs --port 18732 --dir build \
  --session-api-key "$LOCAL_TEST_SESSION_KEY" --disable-telemetry \
  --route /api=http://127.0.0.1:18731 \
  --route /server_info=http://127.0.0.1:18731 \
  --route /sockets=http://127.0.0.1:18731 \
  --route /health=http://127.0.0.1:18731
```

In the real browser, edited the default ACP/Codex profile, clicked **Sign in
with ChatGPT**, displayed/copied the OpenAI user code, cancelled the attempt,
and expanded the manual auth.json fallback. Actual OpenAI device initiation
returned HTTP 200. No page errors occurred. No account authorization or model
call was performed. The captured one-time attempt has been cancelled.

- [Settings screenshot](codex-settings.png)
- [Manual fallback screenshot](codex-fallback.png)
- The actual-device-code video is kept locally and is not published because it
  contains a real one-time authorization code.

The screenshots demonstrate rendering. Real device initiation, copy/cancel
and manual fallback were repeated against the final built Agent Server wheel
after restarting the local server; no page errors occurred. The 46 focused
frontend tests and 81 backend/cross tests also passed again.

## Public fixture video

The public [workflow video](codex-demo-flow.webm) uses the real built Canvas and
the final packaged Agent Server, HTTP routes, CodexAuthService and encrypted
store. Only the OpenAI transport is replaced by a deterministic fixture.
The backend is visibly named **Mock OpenAI transport fixture**; codes are
**DEMO-ONLY-1/2** and every token is a nonfunctional test value. No real account
or OpenAI login credential is used in this recording. It demonstrates pending,
copy/cancel, successful status publication, reload detection and disconnect.
It is evidence of integration behavior, and does not replace human account
authorization or a real Codex model turn.

## Authorized account verification — 2026-10-02

The human completed device authorization in the real Canvas and observed
**Connected to ChatGPT**. The final packaged backend confirmed `connected=true`.
Three isolated conversations launched via the active Codex Agent Profile and
`codex-acp@1.10.0` reached `finished` and returned `OK`: before restart, after
server restart, and after actual OpenAI token refresh. These were real model
requests with the human-authorized subscription, without an API key.

Refresh was triggered by forcing only the expiry predicate in a local harness.
The real OpenAI transport, encrypted store and versioned update were used;
access/refresh tokens changed and the running server remained connected. This
is not a naturally expired-token or prolonged-use test. The observations are
available as [credential-free live results](codex-live-results.json).

Default bare `npx` launch failed with `[WinError 2]` on this Windows host.
The local test profile now invokes installed `node.exe` and npm's `npx-cli.js`
with the same pinned adapter and the existing isolated npm cache. This profile
workaround does not fix the default Windows launcher. Browser reload and real
account disconnect remain pending; the human's connection was kept available.
The public fixture video above still demonstrates fixture behavior only.

## Remaining acceptance and merge gates

- Real account browser reload and disconnect validation.
- Docker and remotely hosted Agent Server validation.
- SDK/Agent Server release, then exact client/lock updates and minimum server
  compatibility version in Canvas. No speculative version bump is committed.
- OpenHands Cloud App API requires a separate user-scoped integration; existing
  fallback behavior remains, and hosted Agent Server URLs are supported by the
  Agent Server contract.
- Human-only PR testing note required by the repository template.
- Remove `.pr/` manually before merging this fork PR.
