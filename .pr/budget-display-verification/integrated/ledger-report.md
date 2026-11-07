# Verification ledger — run budget-display-integrated

Revision `90350be6e173d1a28fbd74038fb3a2ca8c650017`, local mode, Agent Server 1.58.2, base http://127.0.0.1:18800.

Checks: 11 — pass 10, fail 0, blocked 0, not-run 1, xfail 0, xpass 0. Families: 1.

## Families

| Family | pass | fail | blocked | not-run | xfail | xpass |
|---|---|---|---|---|---|---|
| F27 | 10 | 0 | 0 | 1 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F27.budget-planner | local UI switch back to Code after planner | Code mode switch exists → Historical local Canvas gates switch to Cloud; API continuation verified separately | not-run | No mode-state injection or plan-file creation |  |
| F27.budget-hidden | desktop Usage, null cap | Canonical cap paired with this conversation actual spend → Actual cost $0.000157776; no budget meter | pass |  | `unbounded-desktop.png` |
| F27.budget-meter | desktop and phone Usage, positive cap | Canonical cap paired with this conversation actual spend → Actual $0.00255006/$0.05,5.10%; matching original baseline fixture now renders | pass |  | `normal-desktop.png` `normal-phone.png` |
| F27.budget-meter | desktop direct composer live update | Canonical cap paired with this conversation actual spend → Without reload: $0.002709036→$0.002879712,cap$0.05;5.42→5.76% | pass |  | `live-visible.mp4` |
| F27.budget-meter | realUI allfive themes;normal and error fill | Canonical cap paired with this conversation actual spend → Rendered fill/track contrast>=3:1 in allten samples; text contrast notcovered | pass |  | `theme-contrasts.json` |
| F27.budget-overshoot | desktop and phone Usage,positive-cap overshoot | Canonical cap paired with this conversation actual spend → Actual $0.000352608/$0.0001=352.608%;100% geometry; think thenMaxBudgetReached | pass |  | `over-desktop.png` `over-phone.png` |
| F27.budget-planner | Create a plan then plan-mode composer on parent route | Canonical cap paired with this conversation actual spend → Child costs $0.000953496/nullcap; parent remains $0.002879712/$0.05 | pass |  | `planner.mp4` |
| F27.budget-planner | canonical REST parent continuation observed live | Canonical cap paired with this conversation actual spend → API-triggered parent update to$0.003058788/$0.05; childcost unchanged; no UI mode-switch claim | pass |  | `parent-api-followup.mp4` |
| F27.budget-recovery | reload and backend restart then reload and reopen Usage | Canonical cap paired with this conversation actual spend → Same $0.002879712/$0.05 persists; explicit reopen required;11.3s restart wait cut | pass |  | `recovery-visible.mp4` |
| F27.budget-switch | sidebar capped to unbounded and back | Canonical cap paired with this conversation actual spend → Each conversation retains its own actual cost and cap; no stale budget label | pass |  | `switch-readable.mp4` |
| F27.budget-warning | desktop and phone Usage,80–100 percent | Canonical cap paired with this conversation actual spend → Actual $0.00247896/$0.003=82.632%; error fill partially full | pass |  | `warning-desktop.png` `warning-phone.png` |

Artifact paths were rebased to sibling filenames for the published layout; generated results and counts are unchanged.
