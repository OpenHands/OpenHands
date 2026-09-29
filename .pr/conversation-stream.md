# Canvas conversation transport boundary

Before: Canvas constructed WebSocket connections and implemented authentication, retry timers, handshake timeouts, and raw sends.
After: the React hook delegates to `ConversationEventStream` from the SDK. Its `socket` handle is the SDK transport, so existing main/planning sends and connection-state reads also cross the SDK boundary. Endpoint construction uses the SDK helper while Canvas resolves its browser origin/proxy URL.

Validation against the integrated SDK client containing software-agent-sdk #5013:
- 44 focused tests passed, four existing skips: hook lifecycle, replay URLs, main/planning message routing, REST fallback, and architecture guard.
- Type checking, ESLint, and production build passed.
- Real Chromium against the same disposable Docker conversation before and after the change: authenticated as the first frame, received conversation events, no credential in the URL, and no page errors. The before/after screenshots show unchanged UI behavior. See canvas-stream-browser.json.

Reproduce: build the SDK client and Canvas; run the isolated static server with /api and /sockets proxies to an Agent Server. Open an existing conversation using the injected session credential. Verify event streaming, then disconnect/reconnect. The SDK PR separately records a live reconnect and its full 324-test TypeScript suite.

Release dependency: the public package pin remains the existing released version. Keep this PR draft until SDK #5013 is released, then bump to that exact release. Local verification uses the built SDK, not the currently published client. Normal CI against the old package will not resolve the new exports yet.

The earlier Canvas #17387 independently migrates the Git/bash path and scoped runtime UI. This PR targets main because its conversation transport change has no code dependency on that PR. Their composition removes the remaining runtime transports from Canvas.

## CI blocker: the required client API is unpublished (verified 2026-09-29)

Every failing job fails for one reason: `src/hooks/use-websocket.ts` and `src/utils/websocket-url.ts` import `ConversationEventStream`, `ConversationEventStreamOptions`, `ConversationEventStreamState` and `buildConversationEventStreamUrl`, which **no published `@openhands/typescript-client` release has ever exported**. The pinned `1.49.6` is also the latest published version, so this is not a regression in a released artifact - the feature has not shipped yet.

Evidence:
- `dist-tags` is `{"latest":"1.49.6"}`; `.../1.50.0` returns `404` on npmjs.
- Every tarball from `1.45.0` through `1.49.6` was downloaded and grepped: zero dist files reference either symbol.
- The CI `TS2305`/`TS2724` errors and the two `TS7006` implicit-`any` errors in the test file reproduce exactly with the pinned package restored; the `TS7006` errors are downstream of the unresolved type.

The branch code itself is correct. Building the client from `software-agent-sdk` `main` (merged #5013, squash `b1b237ca103`) and overlaying it on the pinned package gives: `npm run typecheck` clean; `use-websocket.test.ts` + `no-direct-agent-server-calls.test.ts` 14 passed / 4 skipped; the broader 7-file WebSocket suite 85 passed / 4 skipped; `npm run build:app` passes; ESLint clean on the changed files. The only delta between red and green CI is the client release.

Adapting to `1.49.6`'s supported API is not an option: the only published transport, `WebSocketCallbackClient`, passes the session key in the URL query string (the credential-in-URL behavior this PR and #16095 removed) and exposes none of `readyState`, `send`, `reconnect` or the lifecycle/state callbacks. Using it would both regress security and reintroduce the Canvas-owned transport that `src/api/no-direct-agent-server-calls.test.ts` guards against. Pinning an unreleased `1.50.0` would break `npm ci`, and `scripts/check-sdk-version-sync.mjs` requires the `package.json` pin to equal `config/defaults.json` `versions.agentServer`.

Dependency order:
1. `software-agent-sdk#5013` - merged to `main`; the exports exist in client source.
2. Release `@openhands/typescript-client@1.50.0` - release PR `software-agent-sdk#5374` is open and `mergeable_state: blocked`, its only failing check being `security-scan` approval drift. Merging it runs `create-release.yml`, which dispatches `typescript-client-npm-publish.yml` to publish to npm. This is the human-gated step.
3. Bump the pin in `package.json` + `package-lock.json` and `config/defaults.json` `versions.agentServer` to `1.50.0` together on this branch.
