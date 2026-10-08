# Verification ledger — run 2026-10-08T2049-d59e2b

Revision `53c8b4d9ca3878a85eba57e9922467d94dddebb0`, local mode, Agent Server 1.53.0, base http://127.0.0.1:18810.

Checks: 2 — pass 2, fail 0, blocked 0, not-run 0. Families: 1.

## Families

| Family | pass | fail | blocked | not-run |
|---|---|---|---|---|
| F05 | 2 | 0 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F05.image-only-send | SDK main stack: image + 'Which two colours ... Do not run any tools.' | reply names orange and blue → user MessageEvent images 1; agent 'Blue and orange.'; 0 actions | pass | launch --sdk-path software-agent-sdk@16662b0 (main, includes #5460/#5467); same build of Canvas 53c8b4d | `evidence/F05.image-only-send/sdk-main-colours.png` |
| F05.image-only-send | SDK main stack: image alone from Home (empty field) | the model receives the picture → user MessageEvent text ' ', images 1; the model's first reasoning: 'four vertical stripes: blue, orange, blue, orange' (no text accompanying it); paused after its first command | pass | on agent-server 1.53.0 the same send reached the model without the image (first thought: 'contains only the system prompt') |  |
