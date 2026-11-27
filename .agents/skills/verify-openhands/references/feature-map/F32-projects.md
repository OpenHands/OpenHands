# F32 — Projects (default view)

A project groups the conversations that work toward one goal. The Agent Server
stores it ([software-agent-sdk#6602](https://github.com/OpenHands/software-agent-sdk/pull/6602)):
the record, a `project` tag on each member, a coordinator conversation that
starts workers, and project files. Canvas shows what the server stores: a
Projects list, a project page with the coordinator, a board grouped by what
needs the user, the files and the project's memory. Inside a project (its page
or one of its conversations) the sidebar lists the project's conversations
instead of all conversations, and a member conversation gets a **Project** tab
beside the chat with the project's files and pull requests. Everything sits
behind the `projects` experimental feature (Settings › Experimental).

Source: `src/routes/projects.tsx`, `src/routes/project.tsx`, `src/routes/project-tab.tsx`, `src/hooks/query/use-conversation-project.ts`, `src/hooks/mutation/use-attach-project-file.ts`, `src/hooks/query/use-pull-request-status.ts`, `src/api/git-service/pull-request-status.api.ts`, `src/utils/project-board-group.ts`, `src/utils/pull-request-attention.ts`, `src/utils/user-activity.ts`, `src/components/features/projects/`, `src/api/projects-service/projects-service.api.ts`, `src/hooks/query/use-projects.ts`, `src/hooks/query/use-project-names.ts`, `src/components/features/conversation/project-badge.tsx`, `src/components/features/chat/project-message-header-line.tsx`, `src/utils/project-message-header.ts`, `src/components/features/sidebar/sidebar-rail-body.tsx`, `src/components/features/sidebar/sidebar-nav-link.tsx`, `src/components/features/chat/chat-add-file-button.tsx`, `src/utils/project-artifact-kind.ts`.

## Sub-features

- `F32.off`: with `projects` off (or an Agent Server without Projects), the rail has no Projects item, conversation cards and headers show no project badge, a conversation tagged `project=<hex>` gets no project sidebar, Project tab or **Project file** item, and `/projects` and `/projects/<id>` say `Projects is turned off on this Agent Server. Turn it on in Settings › Experimental.` with a link to `/settings/experimental`, without calling `/api/projects`.
- `F32.entry`: with `projects` on, the rail shows **Projects** with an always-visible **+**; the item opens `/projects`, the **+** opens New project.
- `F32.create`: New project takes a name, goal, optional worker agent profile and the coordinator's workspace; Create sends one `POST /api/projects` with `coordinator`, and the server's new coordinator conversation exists, idle, tagged `project=<id hex>`.
- `F32.list`: `/projects` lists the backend's projects newest first with name and goal, and an empty backend says `No projects yet.`
- `F32.edit`: the pencil beside the project name (`project-edit`) opens Edit with the name and goal filled in; Save renames the project on the page and on the server. **Delete project** (`edit-project-delete`) asks `Delete "<name>"? Its files and memory are deleted. Its conversations are kept.`; confirming opens `/projects` without it, and its conversations remain (the coordinator loses its project tag).
- `F32.coordinator`: the project page's **Talk to the coordinator** opens the coordinator in the normal chat view; a message sent there reaches it like any conversation.
- `F32.board`: the conversations the coordinator starts appear on the board without a reload, grouped Needs you (attention needs_you + asking), Working, Idle, Resolved; a card that stopped on a question shows that question. A worker that resolves itself when it finishes (`resolved=auto`) moves to Resolved (it may show in Idle briefly first) with `Resolved automatically: <why>` on its card.
- `F32.resolve`: Resolve sets `resolved=user` and keeps the other tags; Reopen removes `resolved` (and an automatic resolve's `resolvenote`); the card moves groups both ways.
- `F32.membership`: only the coordinator adds conversations (the project page has no New conversation button or Add a conversation picker); **Remove** asks `Remove "<title>" from this project? The conversation itself is kept.`, then drops the member's `project` tag (with the board's `turn` and `resolved` tags) and takes its card off the board; the coordinator cannot be removed.
- `F32.files`: Upload adds a file marked "Added by you"; a file a conversation saved reads `Saved by <conversation title>` with a link to that conversation; clicking a text file shows its contents, an image or PDF shows inline, other files say there is no preview; Download saves the file; Delete asks first, then removes it.
- `F32.attach`: in a member conversation, the composer's **+** menu offers **Project file**, which picks one of the project's files and attaches it to the message; it is absent outside a project or with `projects` off.
- `F32.badge`: a member conversation's header shows the project's name as a link to its project page.
- `F32.memory`: the project page's **Memory · N** lists `MEMORY.md` and the topic files the project's conversations wrote (`update_project_memory`), with their age; clicking one shows its text below the list. With none it says `Nothing remembered yet. The project's conversations write here what later ones should know.`
- `F32.sidebar`: on a project's page and in its conversations the sidebar shows **All projects** (to `/projects`), `PROJECT` and the project's name (a link to the project page, plain text on the page itself), the Coordinator pinned on top, then the board's groups with each conversation's state, age and saved-file count (`1 file`). Resolved starts folded unless the open conversation is in it; the open conversation reads `You are here`. On `/projects` and outside a project the normal conversation list returns. The phone drawer shows the same list.
- `F32.project-tab`: a member conversation's right panel has a **Project** tab first. **Files · N** groups the project's files into From agents and From you, each with its source and age; clicking one previews it in place, **Attach** adds it to the composer, and **Upload file** adds a file to the project (download and delete stay on the project page). **PRs** lists the pull requests the project's conversations opened, or `No pull requests yet. When a conversation in this project opens one, it shows here.` At phone width the tab is on the `/conversations/<id>/panel` page.
- `F32.headers`: in the coordinator, the server's `[Update from …]`, `[Sent by …]`, `[Reply in …]`, `[Message from the user in …]` and `[Paused by the user: …]` header lines render as one compact line with a working mention; the user's own bracketed text is untouched.

## How to get to it (user POV)

- Turn on **Settings › Experimental › Projects**, or
  `control-openhands api PATCH /api/settings/experimental/projects --data '{"enabled": true}'`.
- Rail **Projects** (`sidebar-projects-link`) or `/projects`; its **+**
  (`sidebar-new-project`) or `/projects?new=1` for New project.
- A project page is `/projects/<project id hex>`; the badge in a member
  conversation's header (`conversation-project-badge`) leads there too.

## Driving it with control-openhands

Preconditions:

- **Recipes describe expected behavior.** Record actual results in the run's
  ledger; unexecuted branches stay `not-run`.
- A local stack whose Agent Server serves Projects (`POST /api/projects`
  with `coordinator`) and meets the UI's minimum Agent Server version:
  `control-openhands api GET /api/settings/experimental` must list the key
  `projects`. The pinned Agent Server (1.64.1) lists it, so a default
  `control-openhands launch --new` runs every F32 recipe. On an Agent Server
  that does not list it, only `F32.off` can run; record the rest `blocked`.
- Canvas reads the flag through `@openhands/typescript-client`'s
  `listExperimentalFeatures`. A client without it reads every flag as off, so
  F32 shows nothing; record the client version with the run.
- `control-openhands llm preset deepseek` (or another authorized LLM) for
  `F32.board`, `F32.headers` and the coordinator branches; the REST-only
  branches need no model.
- Paper theme: `control-openhands browser eval "localStorage.setItem('openhands-color-theme','paper')"` and reload.

- **Off (`F32.off`).** With `projects` off,
  `control-openhands browser goto /conversations` and
  `control-openhands browser count 'testid=sidebar-projects-link'` → 0.
  `control-openhands browser goto /projects`: `testid=projects-off` is visible
  and links to `/settings/experimental`. A conversation tagged
  `project=<hex>` shows no project chip on its card and no
  `testid=conversation-project-badge` in its header.

- **Entry and create (`F32.entry`, `F32.create`, `F32.list`).** Turn
  `projects` on (`control-openhands browser click 'testid=experimental-switch-projects' --force`
  on `/settings/experimental`) and reload. Before creating anything,
  `/projects` shows `No projects yet.` (`testid=projects-empty`). Click `testid=sidebar-new-project`;
  `testid=new-project-modal` opens over `/projects?new=1`. Fill
  `testid=new-project-name` (`QA F32`), `testid=new-project-goal`, leave the
  worker profile on the default, pick the workspace, click
  `testid=new-project-create`. The URL becomes `/projects/<id>`;
  `control-openhands api GET /api/projects/<id>` has
  `coordinator_conversation_id`, and that conversation is idle with tag
  `project=<id>` and no user message. Back on `/projects`, `testid=project-row`
  lists QA F32 first; a second project created the same way lists above it.

- **Edit and delete (`F32.edit`).** On a second project's page click
  `testid=project-edit`; `browser value 'testid=edit-project-name'` is its
  name. Fill a new name and click `testid=edit-project-save`:
  `testid=project-name` and `api GET /api/projects/<id>` show it. Click
  `testid=project-edit`, then `testid=edit-project-delete`; the confirmation
  reads `Delete "<name>"? Its files and memory are deleted. Its conversations are kept.`
  `testid=confirm-button` opens `/projects` without its row, `api GET
  /api/projects/<id>` is `404`, and its coordinator is still in
  `control-openhands conversation list`.

- **Coordinator and board (`F32.coordinator`, `F32.board`).** Click
  `testid=project-open-coordinator` → `/conversations/<coordinator>`. Send:
  `Start one conversation that writes wordcount.py and one that writes slugify.py, then tell me when both are finished.`
  Return to the project page without reloading (click
  `testid=project-sidebar-name`). Within the poll interval the two workers
  appear in `testid=project-group-working`; when they finish, they move to
  Idle (`testid=project-group-idle`), to `testid=project-group-resolved` with
  `Resolved automatically: …` when they resolved themselves (deepseek-flash
  workers usually do), or to `testid=project-group-needs_you` with the
  question shown if they asked one. Compare with
  `control-openhands api GET /api/projects/<id>/conversations`.

- **Resolve and Reopen (`F32.resolve`).** On an automatically resolved card,
  `testid=project-card-resolve` reads Reopen: it moves the card to Idle and
  removes `resolved` and `resolvenote`. On an open card click
  `testid=project-card-resolve`; the card moves to
  `testid=project-group-resolved` and the conversation's tags gain
  `resolved: user` with the others kept. Click Reopen on it; it moves back and
  `resolved` is gone.

- **Membership (`F32.membership`).** The project page offers no way to start or
  add a conversation: no New conversation button and no Add a conversation
  picker above the board. Ask the coordinator to start a
  worker; it joins the board with `project=<id>`. Remove it with
  `testid=project-card-remove` and confirm with `testid=confirm-button` (the
  dialog says `The conversation itself is kept.`); the card leaves the board,
  `api GET /api/conversations/<id>` still answers `200`, and its tags no
  longer have `project`, `turn` or `resolved`. The coordinator's card
  (`testid=project-coordinator`) has no Remove.

- **Files (`F32.files`).** `control-openhands browser upload 'testid=project-file-input' <plan.md>`:
  a `testid=project-file` row with `data-source=user` and "Added by you";
  clicking the file name shows the text in `testid=project-file-preview`. Ask the
  coordinator `Save a project file named notes.md whose only content is the line: done. Do not run any other tools.`
  (`write_artifact`); its row has `data-source=agent` and `Saved by <title>`
  linking to `/conversations/<that conversation>`. Upload a PNG and a PDF and
  click each file name: `testid=project-file-preview` has `data-kind=image`
  and contains an `img`, and is itself an `iframe` with `data-kind=pdf` (the
  browser's PDF viewer). Any other file (a `.bin`) shows
  `No preview for this kind of file.` `testid=project-file-download` on a row saves
  the file; `testid=project-file-delete` opens the confirmation modal, and
  `testid=confirm-button` removes the row.

- **Attach (`F32.attach`).** In a worker, open the composer's **+** and click
  `testid=add-project-file-button`; `testid=project-file-picker` lists
  `testid=project-file-pick` rows, and picking one attaches it to the message
  (`testid=project-file-picker-cancel` closes it). In a conversation outside
  the project, the item is absent.

- **Memory (`F32.memory`).** A new project's page shows `Memory · 0` and
  the empty text. Ask the coordinator `Add one entry to the project memory saying: the helpers live at the workspace root. Do not run any other tools.`
  (`update_project_memory`); back on the project page `testid=project-memory`
  reads `Memory · 2` with `MEMORY.md` and a topic file
  (`testid=project-memory-file`). `control-openhands browser click 'testid=project-memory-file >> has-text=MEMORY.md >> role=button'`
  and `testid=project-memory-preview` shows its text.

- **Sidebar (`F32.sidebar`).** On the project page `browser text
  'testid=project-sidebar'` starts `PROJECT`, the name and `Coordinator`; in
  the coordinator its row reads `You are here · <state>` and, once it saved a
  file, `1 file`; finished workers that resolved themselves sit under a folded
  `Resolved · N`. `testid=project-sidebar-name` opens the project page and
  **All projects** opens `/projects`, where `testid=project-sidebar` is gone.
  Open a resolved worker after `browser reload`: the Resolved group's toggle
  (`testid=project-panel-group-resolved-toggle`) has `aria-expanded=true` and
  the worker reads `You are here`. At `browser viewport phone`,
  `testid=sidebar-mobile-menu-toggle` opens the drawer with the same list.
  Known failure (reproduced 2026-11-24, [#20030](https://github.com/OpenHands/OpenHands/issues/20030)): opening a resolved worker from the
  coordinator's mention (`testid=project-message-header >> testid=conversation-mention`,
  a client-side navigation) leaves the Resolved group folded
  (`aria-expanded=false`) and no row reads `You are here` until a reload.

- **Project tab (`F32.project-tab`).** In a worker click
  `testid=right-panel-toggle`; `browser snapshot 'testid=tabs-pane-header'`
  lists `tab "Project"` first. Click it: `testid=project-panel` shows
  `testid=project-panel-tab-files` (`Files · N`) with the groups
  `testid=project-panel-group-files-agents` and
  `testid=project-panel-group-files-user`. Click a file's name in
  `testid=project-panel-file` and `testid=project-file-preview` shows it;
  `testid=project-panel-file-attach` puts the file in the composer.
  `browser upload 'testid=project-panel-file-input' <file>` adds a row and
  raises the count. `testid=project-panel-tab-prs` shows
  `testid=project-panel-prs-empty` until a conversation opens a pull request
  (rows `testid=project-panel-pr` need a git provider and a real PR). At
  phone width the panel toggle opens `/conversations/<id>/panel` with the same
  tab. A conversation outside a project has no Project tab.

- **Badge and headers (`F32.badge`, `F32.headers`).** In a worker,
  `testid=conversation-project-badge` shows QA F32 and leads to the project
  page. In the coordinator, each worker update renders as
  `testid=project-message-header` with `data-kind=update_from` and a
  `testid=conversation-mention` that opens the worker; no raw
  `[Update from` text is shown.

## Gotchas

- `GET /api/projects/<id>/artifacts/<name>` answers
  `application/octet-stream` with `Content-Disposition: attachment`; the
  preview fetches it as text and renders it itself. Opening that URL in a tab
  downloads the file, which is expected.

- The board polls every 5 s and the files every 15 s; wait for them rather
  than reloading, which would hide a refresh bug.
- While a conversation is running, the server can return tags from before a
  PATCH; resolve a finished card, as the resolved banner does.
- `testid=project-sidebar` exists twice in a phone layout (the hidden desktop
  rail and the drawer); wait on the drawer's content or screenshot it rather
  than `browser wait 'testid=project-sidebar' --state visible`.
- Deleting a project clears its coordinator's tags; the coordinator stays in
  the conversation list.
- Not mapped yet: the PRs view with real pull requests (their status and
  attention), the `ready-pr` and `needs_you` card states, and the worker
  agent profile choice in New project.
