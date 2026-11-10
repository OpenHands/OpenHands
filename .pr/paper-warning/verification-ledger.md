# Verification ledger — run 2026-11-10T0138-59dd02

Revision `efc1a9cfdac3c26170776160ed6dcbc14d00e6f3`, local mode, Agent Server 1.61.0, base http://127.0.0.1:18800.

Checks: 5 — pass 3, fail 0, blocked 2, not-run 0, xfail 0, xpass 0. Families: 5.

## Families

| Family | pass | fail | blocked | not-run | xfail | xpass |
|---|---|---|---|---|---|---|
| F06 | 0 | 0 | 1 | 0 | 0 | 0 |
| F08 | 0 | 0 | 1 | 0 | 0 | 0 |
| F10 | 1 | 0 | 0 | 0 | 0 | 0 |
| F16 | 1 | 0 | 0 | 0 | 0 | 0 |
| F24 | 1 | 0 | 0 | 0 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F06.markdown-rendering | Conversation Markdown in Paper | Paper Markdown checks → not driven | blocked | Needs a real conversation and model key (no DEEPSEEK_API_KEY in this environment). This change touches only warning and input-border tokens. |  |
| F08.diff-view-modes | Diff view in Paper | Paper diff checks → not driven | blocked | Needs an agent-made change, so a model key (none here). Monaco keeps its own concrete palette; this change does not touch it. |  |
| F10.broken-link | Settings > LLM (profile QA_broken arranged with an orphaned connection id) | Broken link badge; Paper warning label readable on its fill → QA_broken deepseek/deepseek-chat Broken link; badge rgb(255, 250, 242) on rgb(112, 85, 21), 6.74:1 | pass |  | `evidence/F10.broken-link/paper-fix-page.png` |
| F16.paper-theme | Settings > Application > Color Theme > Paper | light paper; rgb(244, 239, 230) \| "Canvas Newsreader", Georgia, serif; phone listbox inside 390px, focus-visible, Escape keeps Paper → light paper; rgb(244, 239, 230) \| "Canvas Newsreader", Georgia, serif; insideViewport true, no horizontal overflow, focus-visible true, Paper kept | pass |  | `evidence/F16.paper-theme/desktop.png` `evidence/F16.paper-theme/phone.png` |
| F24.sync-now | Automations > Git Sync, after Save and sync and one new automation | Pending changes 1 in span.text-warning, readable (4.5:1) → Pending changes 1, text-warning rgb(112, 85, 21) on rgb(236, 229, 218), 5.60:1 | pass | Pending state only; Sync now and later bullets not driven | `evidence/F24.sync-now/paper-fix-pending.png` |
