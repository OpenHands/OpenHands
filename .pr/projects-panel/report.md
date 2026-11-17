# Project views: live run

Agent Server from OpenHands/software-agent-sdk#6922 (on top of #6602), Canvas dev server on this branch, Paper theme, DeepSeek as the model. A coordinator had already started three workers (`gh pr create` was a demo stub printing real PR URLs, and the agents were told so; PR state and checks come from the real GitHub API). In this take a message went to the "Rate-limit login" worker, which really edited the project file.

- `1-coordinator.png`: in the coordinator, the sidebar lists the project's conversations instead of all conversations, under a break from the app navigation and the project's name as their title. The coordinator is pinned and marked "You are here".
- `sidebar-header-before-after.png`: the top of the sidebar before and after that heading.
- `3-working.png`: after the message, the worker's row moves to Working.
- `4-done.png`: it finishes; the row goes back to Idle with the agent's latest line.
- `5-files.png`, `6-prs.png`: the right-side Project tab keeps Files and PRs.
- `7-project-page.png`: on the project's page, the sidebar shows the same list, with the project name marked.
- `8-all.png`: outside a project, the regular conversation list is back.
