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

The original Windows default-launch failure is fixed by SDK commit
`5ab913acbc8d938f21a18e981dd9aa3224eaf2ca`. The packaged SDK launches the installed
npm shim through Node for both cache warming and ACP startup, preserving the
built-in command and arguments. Two native offline launch tests pass.
The local profile's custom command was removed. A new default-preset
conversation and the recovered original failed conversation both returned
`OK`, bringing the real-account total to five successful turns. The actual
Canvas was reloaded, then its default profile showed **Codex**, the default
bare `npx` command and **Connected to ChatGPT**, with blank API fields.
The human remains connected; actual account disconnect and natural expiry
have not been repeated. Public fixture media remain synthetic OAuth evidence.

The isolated hosted Linux runner passed 594 backend/ACP tests (two native
Windows cases skipped) and all 348 TypeScript client tests; all Python
candidate packages and the TypeScript package built successfully.
The canonical Docker image and real container HTTP acceptance also passed.
This is separate from upstream PR CI and official registry publication.


## Docker / hosted Linux acceptance and candidate publication

The [successful cloud-hosted Linux run](https://github.com/luxleader/software-agent-sdk/actions/runs/37023860358) built the canonical Agent Server
`source-minimal` Docker target through the SDK's sdist-based builder, with the
pinned Codex ACP provider. The image ran as its normal non-root user, exposed
only a host-loopback port and used a fresh named volume and test session key.
Real HTTP requests verified session authentication, device pending/success,
credential-free responses, encrypted storage, restart persistence, refresh,
logout deletion and cancellation of an in-flight login. These lifecycle checks
replace only OpenAI's OAuth transport with synthetic tokens. The restarted
unmocked image also initiated and cancelled an actual OpenAI device challenge;
its code and handle were never included in public output. No human account
credentials were sent to the runner and no remote model request was made.
See [credential-free container results](codex-remote-results.json).

Four Python wheels/sdists, the TypeScript tarball, source manifest, checksums
and the acceptance report are published as a [fork candidate prerelease](https://github.com/luxleader/software-agent-sdk/releases/tag/codex-oauth-17372-candidate-5ab913ac).
The changed SDK/server modules match production commit `5ab913ac` (line endings
normalized). Package metadata remains the development version `1.50.1`; this
does not mean that PyPI/npm's `1.50.1` contains this feature. Candidate publication
does not replace the official OpenHands release or downstream exact pins.

## Remaining acceptance and merge gates

- Real account disconnect and natural expiry validation.
- Human authorization and a real model turn on a remotely hosted Agent Server
  remain unverified; hosted Docker/HTTP acceptance passed as scoped above.
- SDK/Agent Server release, then exact client/lock updates and minimum server
  compatibility version in Canvas. No speculative version bump is committed.
- OpenHands Cloud App API requires a separate user-scoped integration; existing
  fallback behavior remains, and hosted Agent Server URLs are supported by the
  Agent Server contract.
- The author authorized an AI-assisted HUMAN summary; assistance is disclosed
  and the latest PR description check passed.
- Remove `.pr/` manually before merging this fork PR.
