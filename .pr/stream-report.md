# Live Canvas: SDK event stream and reconnection

Captured 2026-09-13 15:01–15:02 UTC in the isolated, UI-created factory on Canvas port 9104. [Recording](stream-reconnect.gif) shows the actual reviewer running on [Airbnb PR #46](https://github.com/neubig/airbnb-clone/pull/46), inspecting files and review threads. The browser, model, agent, tools, and server are real.

The browser-only recording proxy forwarded every WebSocket frame unchanged from the actual Agent Server, then closed this tab's first connection once with code 1012. The SDK reconnected automatically; the agent continued working throughout. No application instructions were sent and no factory service was restarted.

[Sanitized transport observations](stream-summary.json) show two connections, each sending authentication as its first client frame, with no credential in either URL. The second connection retains the replay cursor. It received 1,673 frames after reconnection, including actions, observations, state updates, and streaming deltas. Across both connections, 1,911 distinct event IDs were observed. Five IDs were resent at the replay boundary; wire replay is expected and must not be confused with duplicate UI entries. This recording demonstrates reconnection and continued visible work; it does not independently instrument the UI store's deduplication implementation.

## Versions and scope

- SDK TypeScript change: software-agent-sdk #5013, source commit `2e59a53d10d6e181d014091892b7e6cc05660019`.
- Canvas consumer: OpenHands #17389, source commit `e25fe1e72240ff2ef17039edca3af86f804de24d`.
- Compiled integration: Canvas `a3c7915db5f47800f5240a25987b2af75b0d8d04`, using SDK TypeScript integration `ac6d12b0b9d76f1cc38a6eb1ea51cd92e34a0bf2`.
- Host Agent Server process: integration `7fe37443b`; Docker worker image `sha256:d8bbd04fd795e629a8b67975cd3431697ba086588d6cb9a34f43e2d4b3e379d4` built from SDK `e0c6a73752fe`.
- Automation process: `6df90db2` plus SDK client integration `ac6d12b0b9d76f1cc38a6eb1ea51cd92e34a0bf2`; catalog runtime bundle from extensions integration `591ce05`.
- Conversation: `1ab0aee7-8515-4fca-904f-d09811bbf714`.

This is an after-only enhancement demonstration in the composed preview build. It replaces the earlier empty-conversation screenshots as evidence of a working live agent. It does not claim these unreleased dependencies are available in the published installer or that every transport edge case was demonstrated. The first exploratory attempt using browser offline mode did not actually close the WebSocket and is excluded from this evidence.

The GIF uses six original Canvas screenshots at three seconds each. The tab was closed at the end; no comparison services or Docker workers were added. Original screenshots and the recording script remain in `factory-state/evidence/live-delivery` in the demo workspace.
