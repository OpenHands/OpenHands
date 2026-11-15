# PR #18520 focus verification — 2026-10-23

Reviewed exact head: f7caf580638b5f71e9461476f2ca33d7d819af17. Production build from an isolated checkout of this head. No source change for this evidence.

## Real-account evidence

The live-neutral-\* images show the actual signed-in DigitalOcean connection from the existing localhost:8000 stack, read through a same-origin restricted local preview on localhost:19020. The real account currently returned zero agent configs and zero sessions (HTTP 200), so only New Agent and Refresh were present. The actual session key stayed in server memory; it was not injected into HTML or copied into browser storage. The preview rejects foreign-origin requests, lifecycle RPCs, every /api path, and all WebSockets.

## Fixture-backed running Canvas evidence

The fixture-\* images show the unchanged exact-head Canvas production build and its real MARS API/bridge code against the repository DigitalOcean stand-in from #18714. Only DigitalOcean data is a fixture: standin-agent with running and paused sessions. No React hooks, DOM contents, CSS, or browser network responses were mocked or modified. Placeholder token is in-memory and unusable at DigitalOcean. No tunnel, VM, guest server, or LLM was started. The preview allows only read-only list/auth RPCs.

Theme preference is arranged before application boot via the app's documented openhands-color-theme localStorage preference, then Canvas's own bootstrap/theme code applies it. This is a theme fixture precondition, not evidence of changing themes through Settings. Settings is inaccessible with no connected backend; an attempted disposable local-key connection was denied by automatic approval review before submission and was abandoned. No real credential was submitted.

Five built-in themes × desktop1470x761 and phone390x844 × four keyboard-focus targets = 40 captures. Real Tab presses placed focus on New Agent, Refresh agents, agent-open, and nested-session controls; computed-focus.json records :focus-visible, outline geometry/colors, backgrounds, bounds, and contrast. Every ring is 2px solid; +2px offset on New Agent/Refresh and -2px inset on rows. All fit the phone viewport. Focus contrast against the rendered panel surface: neutral16.29:1, Deepsea15.32:1, Neo16.29:1, Light+4.06:1, Solarized Light3.54:1.

Existing light-theme white title/row text is visible in these images and is outside this four-class diff. This focus-only PR does not change that text. No action/launch was activated. Viewport override was reset after capture.

The full measurement matrix is [computed-focus.json](computed-focus.json). The checked-in set selects twelve representative captures from the forty-four original browser screenshots; all are unedited. All forty fixture cases were measured and captured, and the complete original capture set remains in the local review artifact directory.

The fixture source is the repository DigitalOcean stand-in from #18714, commit de2101d701, with its real `createStandin` HTTP service. This PR’s `createMarsTunnelBridge` and `createMarsWebBridge` connected to it using its placeholder PAT in process memory. A temporary read-only static wrapper served the built Canvas, forwarded no Agent Server APIs, and refused lifecycle methods and WebSockets. This is appearance evidence, not live DigitalOcean lifecycle or Settings UI verification.

## Representative captures

| Theme / source          | Desktop                                                     | Phone                                                |
| ----------------------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| Real account, neutral   | [Desktop](live-neutral-desktop-new-agent.png)               | [Phone](live-neutral-phone-refresh.png)              |
| Neutral fixture         | [Desktop](fixture-openhands-neutral-desktop-agent-open.png) | [Phone](fixture-openhands-neutral-phone-session.png) |
| Deepsea fixture         | [Desktop](fixture-openhands-deepsea-desktop-new-agent.png)  | [Phone](fixture-openhands-deepsea-phone-refresh.png) |
| Neo fixture             | [Desktop](fixture-openhands-neo-desktop-refresh.png)        | [Phone](fixture-openhands-neo-phone-agent-open.png)  |
| Light+ fixture          | [Desktop](fixture-light-plus-desktop-new-agent.png)         | [Phone](fixture-light-plus-phone-session.png)        |
| Solarized Light fixture | [Desktop](fixture-solarized-light-desktop-agent-open.png)   | [Phone](fixture-solarized-light-phone-new-agent.png) |
