# Reference material

Inputs for revising `../index.html`. Copied sources keep a `.txt` suffix so CI, linters and test discovery ignore them.
Nothing here is imported or run.

| Folder | What | Pinned to |
| --- | --- | --- |
| `automation/` | Outcome store in `OpenHands/automation`: schemas, value storage, aggregation, read routes, migration, verification notes. `PRS.md` lists the related PRs and their state. | `OpenHands/automation` `main` at `3a913056fbed96b12e57d6abb5e5e5783fc81965` (2026-11-25) |
| `prototype/` | Factories prototype UI the cards must match: shared card components, Overview page, flow row, design tokens. | `DevinVinson/OpenHands` branch `dv/openhands-factories-prototype-a9220b` at `f3c2618957c8b0fd02397e9f06c78cad5f6de814` |
| `notion/` | Links and a short digest of the four Notion pages the design drew on. The pages stay in Notion. | Notion, read 2026-11-24 |

To see the prototype running, check out the branch above, then `npm install`, build with the mock API and serve it
(`scripts/dev-static.mjs`). The Factories routes are `/factories` and `/factories/:id/*`.
