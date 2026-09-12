# Canvas workspace boundary evidence

Base: `28464621d` (`main`). Both builds use the TypeScript client compiled from the SDK runtime integration; SDK #4966/#3403 must release that client before this consumer can merge.

## Real browser reproduction

1. Build both base and PR with `npm run build`.
2. Run `node scripts/static-server.mjs --host 127.0.0.1 --port <port> --dir build` with the launcher session credential and proxy routes to a Docker Agent Server. Do not publish the credential.
3. Open the home page in Chromium and select **Open Workspace**.

Observed on base: the button is enabled and opens a host workspace dialog against a Docker backend. On the PR: the button is disabled and the home page says **New isolated workspace**. Both pages produced zero browser page errors.

![Base allows host workspace selection](workspace-baseline.png)
![PR advertises isolated workspace and disables host selection](workspace-pr.png)

## Regression coverage

- 125 focused API/runtime-boundary tests passed against the integrated TypeScript client (no capability skips).
- 24 workspace-selection/menu UI tests passed after translations were updated.
- `npm run typecheck` and `npm run build` passed.
- Targeted ESLint passed.

API tests exercise actual typed clients against HTTP fixtures: file, Git, terminal, and VS Code calls retain conversation scope. They cover isolated workspace creation, rejecting host project selection, preserving local defaults, and suppressing host project hooks in isolated conversations.

Live visual evidence above covers the home controls. Full interactive terminal/file/VS Code verification against a Docker conversation remains required before promoting this draft for review.
