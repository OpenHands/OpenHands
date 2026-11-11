# Verification ledger — run 2026-11-10T0228-cfa723

Revision `936f3ed8c486ba4bbdb26f817e7da5e805ab735b`, local mode, Agent Server 1.61.0, base http://127.0.0.1:18800.

Checks: 6 — pass 4, fail 0, blocked 2, not-run 0, xfail 0, xpass 0. Families: 6.

## Families

| Family | pass | fail | blocked | not-run | xfail | xpass |
|---|---|---|---|---|---|---|
| F06 | 1 | 0 | 0 | 0 | 0 | 0 |
| F08 | 0 | 0 | 1 | 0 | 0 | 0 |
| F10 | 1 | 0 | 0 | 0 | 0 | 0 |
| F16 | 1 | 0 | 0 | 0 | 0 | 0 |
| F24 | 1 | 0 | 0 | 0 | 0 | 0 |
| F27 | 0 | 0 | 1 | 0 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F08.diff-view-modes | Changes panel diff in Paper | Monaco Paper inks → not driven | blocked | Needs an agent-made workspace change, so a model key (none here). Monaco palette contrast is covered by the theme tests. |  |
| F27.terminal-output | Terminal in Paper | ANSI inks → not driven | blocked | Needs genuine terminal events from a running agent. ANSI contrast is covered by terminal-themes.test.ts on base and surface. |  |
| F06.markdown-rendering | Conversation user message (run:false, unreachable model endpoint, never run) | Newsreader heading; ink-weight code; readable link and alert inks → heading Canvas Newsreader; comment rgb(97, 84, 72); def rgb(36, 28, 23) bold; string rgb(69, 87, 47); link rgb(154, 38, 22); conversation idle | pass |  | `evidence/F06.markdown-rendering/after-message.png` `evidence/F06.markdown-rendering/after-message-code.png` |
| F10.broken-link | Settings > LLM, QA_broken arranged with an orphaned connection id | Broken link badge readable → light text on dark ochre, 6.65:1 | pass |  | `evidence/F10.broken-link/after-llm.png` |
| F16.paper-theme | Settings > Application > Color Theme > Paper | light paper; aged page; primary prints in reverse; standalone links visible → light paper; body rgb(243, 231, 211); Save rgb(36, 28, 23) / rgb(248, 240, 225) 14.8:1; Manage LLM profiles rgb(154, 38, 22) | pass | F16 recipe values (rgb(244, 239, 230), rust primary) describe the old Paper; map update belongs in a separate verify-skill PR | `evidence/F16.paper-theme/after-settings.png` `evidence/F16.paper-theme/after-home.png` |
| F24.sync-now | Automations > Git Sync after Save and sync and one new automation | Pending changes count readable (4.5:1) → Pending changes 1 in rgb(114, 77, 22), 5.58:1 on the card | pass | Pending state only | `evidence/F24.sync-now/after-gitsync.png` |
