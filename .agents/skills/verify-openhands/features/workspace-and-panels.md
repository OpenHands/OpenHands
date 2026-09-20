# F03 — Workspace files and conversation panels

Inspect real files, changes and execution evidence without losing state during panel
transitions. Source: [files](../../../../src/components/features/files-tab),
[diff viewer](../../../../src/components/features/diff-viewer),
[terminal hook](../../../../src/hooks/use-terminal.ts),
[panel route](../../../../src/routes/conversation-panel.tsx).

## Sub-features

- `F03.files`: file navigation, source/Markdown views, upload/download and editor handoff.
- `F03.git`: changed files, expanded diffs and commit history.
- `F03.terminal`: real output, reload/history restoration and later append.
- `F03.browser`: actual browser-tool output, not only its empty state.
- `F03.planner-tasks-usage`: populated plan, task transitions, overview and usage values.
- `F03.transitions`: panel open/close/switch and standalone panel entry, without exceptions.

## How to get to it (user POV)

Open a workspace-backed conversation; use the right panel toggle and its Files,
Changes/commits, Terminal, Browser, Planner, Tasks and usage/overview controls.
Exercise `/conversations/:conversationId/panel` separately when it is in scope.

## Driving it with Playwright

Preconditions: F02 real tool events; disposable git repo with an initial commit,
tracked modification, Markdown file and `.agents_tmp/PLAN.md`. Browser/VS Code
checks need those runtime capabilities, not substituted screenshots.

- **Open:** inspect `right-panel-toggle`'s `aria-pressed` before clicking. Use
  `conversation-tab-files`, `conversation-tab-commits` or the overflow menu
  `conversation-tabs-menu-open-commits`; inspect current handles for other tabs.
- **Files/Git:** select `file-quick-row`; compare source and rendered content to disk.
  Use `files-tab-content-mode-toggle` / `files-tab-diff-toggle` where offered. Expand
  a changed-file diff and check its actual additions/removals and real commit rows.
  Upload a disposable file and verify its workspace path; verify any download bytes.
- **Terminal:** issue a harmless real command through chat, inspect its output here,
  reload, and verify it remains. Run another command and check append/no duplication.
- **Planner/tasks/usage:** populate the real plan file and ask the agent to use its
  task tracker. Check the plan, task statuses and actual usage—not hardcoded UI data.
- **Browser/editor:** run a permitted browser-tool visit to a harmless local page and
  inspect the result; open the workspace in the provided editor if available.
- **Transitions:** repeatedly switch expanded diff → Planner → Files, capture
  `pageerror` and ensure views still render. Compare phone/desktop panel controls.

## Gotchas

Clicking an already selected conversation tab can close the drawer. Mounted
controls can remain offscreen. Planner reads `.agents_tmp/PLAN.md`, not `PLAN.md`.
Recheck [#17566](https://github.com/OpenHands/OpenHands/issues/17566) and
[#17567](https://github.com/OpenHands/OpenHands/issues/17567); a rendered destination
panel is not a pass if Monaco throws during the transition.
