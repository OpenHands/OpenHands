# Evidence — attachment upload 422 before/after (PR #18426)

Scenario that reproduces the natural divergence: attach a folder (/tmp/hostws) via the
home "Open Workspace" launcher, switch workspace mode to **new_worktree**, then send a
`.zip` attachment. The agent-server stores `working_dir=/tmp/conversation-worktrees/<id>/hostws`
but Canvas stores `selected_workspace=/tmp/hostws`.

## BEFORE (origin/main `src/api/conversation-file-upload.api.ts`)

- create POST /api/conversations -> working_dir `/tmp/conversation-worktrees/<id>/hostws`
- upload  POST .../file/upload?path=**/tmp/hostws**/attach-test.zip
  -> **HTTP 422** `{"detail":"path must be inside the conversation workspace"}`
- toast shown: `HTTP request failed (422 ): {"detail":"path must be inside the conversation workspace"}`
- workspace worktree contains only README.md + .git (attachment NOT delivered)

## AFTER (PR #18426 head)

- create POST /api/conversations -> working_dir `/tmp/conversation-worktrees/<id>/hostws`
- upload  POST .../file/upload?path=**/tmp/conversation-worktrees/<id>/hostws**/attach-test.zip
  -> **HTTP 200** `{"success":true}`
- no error toast
- workspace worktree contains README.md + .git + **attach-test.zip**

## How the before/after was produced

`node .pr/ui-home-flow.cjs <screenshot> <label>` drives the real UI (Playwright) through
the home launcher. The "before" run temporarily swaps in
`git show origin/main:src/api/conversation-file-upload.api.ts`; the "after" run uses the PR
head. Composite image: `.pr/upload-422-before-after.png`.
