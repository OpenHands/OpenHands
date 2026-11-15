# Live verification: decision cards (`ask_user`)

Both servers were real: Canvas from this branch (on #19503), against the Agent
Server from OpenHands/software-agent-sdk#6852 at `9b42766` (`--sdk-path`), with
deepseek-flash, in the Paper theme. The flag was turned on and off through
Settings › Experimental.

Every conversation got the same task, in a folder holding only a README:

> I need a short config file for this notes app. Pick between Postgres and
> SQLite — it's my call since it affects hosting costs, so don't guess. Then
> write config.yaml.

Checks: 5 — pass 5, fail 0, blocked 0, not-run 0.

| Check | What happened |
|---|---|
| `ask_user.flag` | "Decision cards" starts off; one click and the API reports `enabled: true`. |
| `ask_user.card` | The agent read the folder, said it would ask rather than assume, and called `ask_user` on its own. The card shows the question, SQLite (Recommended) and Postgres with their costs, a note field and "Or answer in your own words in the chat." The run stopped `finished`. |
| `ask_user.tap` | Note "Read the URL from DATABASE_URL.", then a tap on Postgres. The server recorded an `AskUserAnswerEvent` and the answer message; the agent wrote `driver: postgres`, `url: ${DATABASE_URL}`. The card folded to "✓ Postgres" and the note, and the raw `[Answer to your question …]` bubble is not shown, also after a reload. |
| `ask_user.chat` | A second conversation asked again; instead of tapping I wrote in the composer: "Neither for now: write it for SQLite, but keep the file path in an env var NOTES_DB_PATH." The card turned to "Answered in the chat.", my message stayed as a normal bubble, the agent wrote the SQLite config, and there were no error events. |
| `ask_user.off` | With the flag off, a new conversation has no `ask_user` tool: the agent asked the same thing in prose and no card appeared. The first conversation, reopened, renders as it would without this PR (a plain "Ask user …" tool row and the answer message as a bubble). `POST …/ask_user/answer` answers 409 "Decision cards are turned off". |

Screenshots: `flag-on.png`, `asked.png`, `tapped.png`, `answered-in-chat.png`,
`flag-off-new.png`, `flag-off-old.png`.
