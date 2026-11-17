# Project tab: live run

Agent Server from OpenHands/software-agent-sdk#6922 (on top of #6602), Canvas dev server on this branch, Paper theme, DeepSeek as the model.

A coordinator started three workers. `gh pr create` was a demo stub that printed real PR URLs (the agents were told up front it was a stub); every other `gh` call went to the real CLI, and the PR state and checks come from the real GitHub API through the Agent Server's PR route.

- `final-1-conversations.png`: Conversations view. The coordinator is pinned with "You are here"; workers grouped like the board, Resolved folded; PR chips with CI marks and a file count.
- `final-2-files-preview.png`: Files view, "From agents" and "From you", a text file previewed in place.
- `final-3-attached.png`: the file attached to the message being written.
- `final-4-prs.png`: PRs view. Open (#6922, failing CI) and Merged (#6910), each with the conversation that opened it.
- `final-5-worker.png`: after opening a worker from the panel, the panel stays on the Project tab.
