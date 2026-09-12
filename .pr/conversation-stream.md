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
