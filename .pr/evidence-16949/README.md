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

## Full-suite follow-up

The full run at `e0cc9f152be44737ed2aa707536cffa65c8a6652` reported 9,681 passed, 58 failed and 45 pending. Beyond the original 52 main failures, it exposed one missing legacy single-object normalization case and two existing tests that asserted retained stale data or an empty cached list during remount. Normalization is restored; those two expectations now assert current query data. The changed hook, existing ordering suite and Files suite pass all 24 cases.

The other three full-run failures were 30-second timeouts in unrelated lint-policy and library-import tests. Both lint-policy suites passed on a focused rerun. The library-import timeout reproduced on unchanged main with the same 30-second limit; no timeout or unrelated test was changed. The six-file follow-up had 37 passed and that one timeout; the unchanged-main control had 13 passed and the same timeout. See `validation-followup.json`. This is not a green full-suite result.
