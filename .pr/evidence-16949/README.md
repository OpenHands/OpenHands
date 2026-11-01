# Current Git changes after refetch

Code: `71e3fcacbc3d792bdda3784bb41e79eeedd2a611`.
Baseline: upstream main `7df1e5c14c1ff1a9962562b29af0a9748735b9ac`.

## Real-app reproduction

The fixture is a real Git repository on the Agent Server host. Its tracked `sample.txt` was modified before loading the conversation's Commits tab and expanding Uncommitted.

The file was then deleted on disk while the tab remained mounted. After the actual five-minute query stale time, a browser focus/visibility event triggered the existing refetch path. The browser captured a successful 200 response containing `{"status":"DELETED","path":"sample.txt"}`.

- On main, the row retained the Modified (`LuFileDiff`) icon.
- With the fix, the same sequence updated the row to the Deleted (`LuFileMinus`) icon.

The actual response and rendered SVG are recorded in `main-refresh.txt` and `fixed-refresh.txt`. The adjacent screenshots show the running Canvas; no API response or query-cache data was replaced. The browser was driven with Playwright CLI, including the visibility event used to emulate focus in headless Chrome.

Current main already uses the `workspace-files` invalidation prefix for Cloud Files. This patch addresses the remaining Changes object-retention bug, rendered in the current Commits tab.

## Checks

- Four regressions failed against the original hook: M → D, U → A, current objects with stable ordering/pruning, and conversation change.
- The fixed hook has five behavioral cases, including fetch failure/recovery and an empty response. Together with existing Files coverage, 12 tests passed.
- `npm run lint`, `npm run build`, and `npm run build:lib` passed.
- The companion Windows baseline investigation reproduced 52 pre-existing full-suite failures on unchanged main. No verification skill or unrelated platform script was changed.
