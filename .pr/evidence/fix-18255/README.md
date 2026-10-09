# Evidence for the #18255 fix

Four isolated real stacks launched with `verify-openhands`
(`control-openhands launch --new --build always`, then `doctor` and `onboard --skip`):
Agent Server/SDK 1.53.0, automation 1.19.0, fresh state, English,
OpenHands-Neutral, no LLM calls.

| Run | Revision | What it is |
| --- | --- | --- |
| `main-clean` | `4a8f77e` | `main` |
| `main-fix` | `adcdaee` | this branch's fix on `main` |
| `pr-repro` | `d84357b` | #18200 head |
| `pr-fix` | `c9d32fe` | #18200 head + this fix (local cherry-pick, not pushed) |

`measure.sh <run-id> <repo>` ran the same probe on each run. `measurements.jsonl`
holds the raw output. Viewports: desktop 1440 × 1000, tablet 820 × 1180,
phone 390 × 844.

## Results

`<main>`'s bottom edge and which element scrolls:

| Viewport, route | `main` | `main` + fix | #18200 | #18200 + fix |
| --- | --- | --- | --- | --- |
| tablet `/settings/secrets` | 302 of 1180 | 1180 of 1180 | 370 of 1180 | 1180 of 1180 |
| tablet `/settings/app` | outer scrolls | `<main>` scrolls | outer scrolls | `<main>` scrolls |
| phone `/settings/secrets` | 362 of 844 | 844 of 844 | 362 of 844 | 844 of 844 |
| phone `/settings/app` | outer scrolls | `<main>` scrolls | outer scrolls | `<main>` scrolls |
| desktop `/settings/secrets` | 282 of 1000 | 282 of 1000 | 282 of 1000 | 282 of 1000 |
| desktop `/settings/app` | outer scrolls, nav pinned | same | same | same |

Tablet selector opened on short pages (#18200 only; list bottom vs. `<main>` bottom):

| Route | #18200 | #18200 + fix |
| --- | --- | --- |
| `/settings/secrets` | 394 vs 370, clipped | 394 vs 1180, full list |
| `/settings/agents` | 394 vs 354, clipped | 394 vs 1180, full list |

`selector-before-after.png` shows those four states.

All four runs: the phone hub → Secrets → Back returns to the hub; re-entering
Application at tablet width (gear → hub → Application) starts at the top; zero
app errors.

## Same screens on `main`

Screenshots of `/settings/secrets` and `/settings/app` from `main-clean` and
`main-fix` at all three viewports are pixel-identical at tablet and phone. At
desktop the only differing pixels (17 × 10) are the backend port in the "synced
from Local backend (http://127.0.0.1:188x0)" line, because the two stacks ran
on different ports.

## Checksums (SHA-256)

```
1c84dbdff289995de7a0184fddff119add092055e76062066330c897bccdfac9  selector-before-after.png
9d15e7db7f33e36e57143ac901b6b28bd9e49d1cc1d5f6c101535551791b7838  measurements.jsonl
```
