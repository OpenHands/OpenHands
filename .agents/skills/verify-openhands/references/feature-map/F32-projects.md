# F32 — Projects (default view)

A project groups the conversations that work toward one goal. The Agent Server
stores it ([software-agent-sdk#6602](https://github.com/OpenHands/software-agent-sdk/pull/6602)):
the record, a `project` tag on each member, a coordinator conversation that
starts workers, and project files. Canvas shows what the server stores: a
Projects list, a project page with the coordinator, a board grouped by what
needs the user, and the files. Everything sits behind the `projects`
experimental feature (Settings › Experimental). Insider's Cat persona,
docked chat and Voice are App features (F30), not this family.

Source: `src/routes/projects.tsx`, `src/routes/project.tsx`, `src/components/features/projects/`, `src/api/projects-service/projects-service.api.ts`, `src/hooks/query/use-projects.ts`, `src/hooks/query/use-project-names.ts`, `src/components/features/conversation/project-badge.tsx`, `src/components/features/chat/project-message-header-line.tsx`, `src/utils/project-message-header.ts`, `src/components/features/sidebar/sidebar-rail-body.tsx`, `src/components/features/sidebar/sidebar-nav-link.tsx`.

## Sub-features

- `F32.off`: with `projects` off, the rail has no Projects item, conversation cards and headers show no project badge, and `/projects` and `/projects/<id>` say how to turn it on without calling `/api/projects`.
- `F32.entry`: with `projects` on, the rail shows **Projects** with an always-visible **+**; the item opens `/projects`, the **+** opens New project.
- `F32.create`: New project takes a name, goal, optional worker agent profile and the coordinator's workspace; Create sends one `POST /api/projects` with `coordinator`, and the server's new coordinator conversation exists, idle, tagged `project=<id hex>`.
- `F32.list`: `/projects` lists the backend's projects newest first with name and goal, and an empty backend says so.
- `F32.coordinator`: the project page's **Talk to the coordinator** opens the coordinator in the normal chat view; a message sent there reaches it like any conversation.
- `F32.board`: the conversations the coordinator starts appear on the board without a reload, grouped Needs you (needs_you + needs_input), Working, Done (finished + idle), Resolved; a card that stopped on a question shows that question.
- `F32.resolve`: Resolve sets `resolved=user` and keeps the other tags; Reopen removes `resolved`; the card moves groups both ways.
- `F32.membership`: **Add a conversation…** adds a recent non-member; **Remove** drops a member's `project` tag; the coordinator cannot be removed.
- `F32.files`: Upload adds a file marked "Added by you"; a file a conversation saved names that conversation; clicking a text file shows its contents, other files say there is no preview.
- `F32.badge`: a member conversation's header shows the project's name as a link to its project page.
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
- A local stack whose Agent Server includes #6602 at or after `73513fdee`
  (`POST /api/projects` with `coordinator`), launched with `--sdk-path` or
  `--sdk-ref`. The released 1.60.x server has no Projects.
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
  `projects` on and reload. Click `testid=sidebar-new-project`;
  `testid=new-project-modal` opens over `/projects?new=1`. Fill
  `testid=new-project-name` (`QA F32`), `testid=new-project-goal`, leave the
  worker profile on the default, pick the workspace, click
  `testid=new-project-create`. The URL becomes `/projects/<id>`;
  `control-openhands api GET /api/projects/<id>` has
  `coordinator_conversation_id`, and that conversation is idle with tag
  `project=<id>` and no user message. Back on `/projects`, `testid=project-row`
  lists QA F32 first.

- **Coordinator and board (`F32.coordinator`, `F32.board`).** Click
  `testid=project-open-coordinator` → `/conversations/<coordinator>`. Send:
  `Start one conversation that writes wordcount.py and one that writes slugify.py, then tell me when both are finished.`
  Return to the project page without reloading. Within the poll interval the
  two workers appear in `testid=project-group-working`; when they finish, they
  move to Done, or to `testid=project-group-needs_you` with the question shown
  if they asked one. Compare with
  `control-openhands api GET /api/projects/<id>/conversations`.

- **Resolve and Reopen (`F32.resolve`).** Click a card's
  `testid=project-card-resolve`; the card moves to
  `testid=project-group-resolved` and the conversation's tags gain
  `resolved: user` with the others kept. Click Reopen on it; it moves back and
  `resolved` is gone.

- **Membership (`F32.membership`).** Start an unrelated conversation, choose it
  in `testid=project-add-conversation` and Add; it joins the board and gains
  `project=<id>`. Remove it with `testid=project-card-remove`; the tag is gone
  and the conversation still exists.

- **Files (`F32.files`).** `control-openhands browser upload 'testid=project-file-input' <plan.md>`:
  a `testid=project-file` row with `data-source=user` and "Added by you", and
  `testid=project-file-preview` shows the text. Ask the coordinator to save a
  file to the project (`write_artifact`); its row has
  `data-source=agent` and names the worker that saved it.

- **Badge and headers (`F32.badge`, `F32.headers`).** In a worker,
  `testid=conversation-project-badge` shows QA F32 and leads to the project
  page. In the coordinator, each worker update renders as
  `testid=project-message-header` with `data-kind=update_from` and a
  `testid=conversation-mention` that opens the worker; no raw
  `[Update from` text is shown.

Evidence (2026-11-09, Canvas #19495 at 53124e926, Agent Server #6602 at
73513fdee, DeepSeek, Paper): screenshots in
[`.pr/projects-default-ui`](https://github.com/OpenHands/OpenHands/tree/53124e9269a94ef62ede5ca9b9c69b23608258e5/.pr/projects-default-ui)
on #19495 cover `F32.off`, `F32.entry`, `F32.create`, `F32.coordinator`,
`F32.board`, `F32.resolve`, `F32.files` (upload), `F32.badge` and
`F32.headers`. `F32.membership` and agent-saved files were not driven.

## Gotchas

- Since #6602 `3404420ea`, `GET /api/projects/<id>/artifacts/<name>` answers
  `application/octet-stream` with `Content-Disposition: attachment`; the
  preview fetches it as text and renders it itself. Opening that URL in a tab
  downloads the file, which is expected.

- The board polls every 5 s and the files every 15 s; wait for them rather
  than reloading, which would hide a refresh bug.
- While a conversation is running, the server can return tags from before a
  PATCH; resolve a finished card, as the resolved banner does.
- The Insider App's Projects page (F30) is a different surface at
  `/extensions/insider-cat/projects`; do not mix its evidence into F32.
