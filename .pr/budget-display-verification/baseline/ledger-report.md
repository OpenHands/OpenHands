# Verification ledger — run budget-display-baseline

Revision `50ae78643fb497b028bdddf6adfd0dea366b5b49`, local mode, Agent Server 1.58.2, base http://127.0.0.1:18800.

Checks: 1 — pass 0, fail 1, blocked 0, not-run 0, xfail 0, xpass 0. Families: 1.

## Families

| Family | pass | fail | blocked | not-run | xfail | xpass |
|---|---|---|---|---|---|---|
| F27 | 0 | 1 | 0 | 0 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F27.budget-meter | Usage desktop and phone, two fresh capped conversations | Canonical positive conversation cap is displayed with actual cost → REST cap0.05; real cost shown; budget meter absent in both fresh conversations | fail | Expected integration gap at exact PR18533 before connecting canonical SDK field; this is not a new color regression. | `evidence/F27.budget-meter/desktop-cap-hidden.png` `evidence/F27.budget-meter/phone-cap-hidden.png` `evidence/F27.budget-meter/repeat-cap-hidden.png` |
