# Live before/after: Cloud child conversation link

Setup: local Agent Canvas (`npm run dev:minimal`, agent-server 1.50.1) with the OpenHands
Enterprise beta instance registered as a Cloud backend. The local parent agent calls
`launch_child_conversation` with `target: "cloud"`; the child is created on the beta instance.

| | Branch | Child conversation | Reported `url` | What the URL serves |
|---|---|---|---|---|
| Before | `main` @ `a02d755` | `6743c69c804f409ea25a623b2a5d5c9b` | `https://app.beta.staging.all-hands-testing.dev/conversations/6743c69c…` | legacy app (0 `/canvas/assets/` refs) |
| After | `fix/cloud-child-conversation-canvas-link` @ `8368e57` | `03832d3e7cd6449eafbde40ba806523b` | `https://app.beta.staging.all-hands-testing.dev/canvas/conversations/03832d3e…` | Agent Canvas (4 `/canvas/assets/` refs) |

Both children were confirmed on the beta instance via `GET /api/v1/app-conversations`
(`execution_status: finished`), then their sandboxes were paused.

- `before-main.png`: the parent quotes the legacy `/conversations/<id>` link.
- `after-fix.png`: the parent quotes the `/canvas/conversations/<id>` link.
