# #17899 before/after on a live Canvas stack

`run17899.sh` here is the exact script. Each run: `control-openhands launch --new --build always`
(private HOME, so no `~/.gitconfig`), `onboard --skip`, an LLM profile pointing at the repo's
mock LLM server (`tests/e2e/mock-llm/scripts/mock-llm-server.py`) scripted to run one terminal
command, then Settings > Application through the UI: Git username `QA Tester`, email
`qa-tester@example.com`, Save. Then `conversation start` through the composer.

The scripted command:

    env | grep '^GIT_' | sort; git init -q qa_t && cd qa_t && git commit --allow-empty -qm x; git log -1 --format='AUTHOR=%an <%ae> COMMITTER=%cn <%ce>'

| | main (`src/api/agent-server-adapter.ts` stashed) | this branch |
|---|---|---|
| GIT_* in the terminal | none | `GIT_AUTHOR_NAME/EMAIL`, `GIT_COMMITTER_NAME/EMAIL` |
| commit | `Author identity unknown`, exit 128 | `AUTHOR=QA Tester <qa-tester@example.com> COMMITTER=QA Tester <qa-tester@example.com>` |
| stored agent tools | server default | `terminal` with `params.env` set, `file_editor`, `task_tracker` |

The commit is made in a repository the agent creates (`qa_t`), so the identity is not tied to the
workspace's own `.git/config`.
