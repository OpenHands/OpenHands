# Issue #16910 production benchmark

The checked-in Playwright fixture mounts the same deterministic histories in
three fresh Chromium contexts: a 5-message × 10-line control and a 20-message
× 50-line code-heavy history. Each context measures one plain-text durable
append and five separately committed streaming deltas.

The wall-clock assertions are opt-in so unrelated parallel E2E tests and CI
worker contention cannot create a false regression. Run them with
`RUN_CONVERSATION_MARKDOWN_PERF=1` in an isolated Playwright invocation.

## Environment

- Windows 11 10.0.26200
- Node.js 24.18.0
- npm 11.16.0
- Playwright 1.62.1
- Google Chrome 151.0.7922.174
- Production `build:mock` bundle served by `sirv-cli`

As described in the issue, the production measurement temporarily exposed the
existing event-store fixture seam in `VITE_MOCK_API` builds. That harness-only
change is not part of the PR.

## Same-machine A/B

The before bundle disables only the three memo boundaries introduced by the
fix. The fixture, browser, stable action lookup, machine, and build mode are
otherwise identical. The after bundle is the submitted implementation.

| Metric                        | Before: control | Before: long history | After: control | After: long history |
| ----------------------------- | --------------: | -------------------: | -------------: | ------------------: |
| Plain append median           |         28.3 ms |             285.8 ms |         7.5 ms |             14.9 ms |
| Streaming median (15 updates) |         25.6 ms |             275.0 ms |         6.4 ms |              6.5 ms |
| Long/control append ratio     |               — |               10.10× |              — |               1.99× |
| Long/control streaming ratio  |               — |               10.74× |              — |               1.02× |
| Maximum long-history update   |               — |             296.6 ms |              — |             15.6 ms |
| Maximum long task             |               — |               296 ms |              — |                0 ms |
| Historical Prism token spans  |             850 |               17,000 |            850 |              17,000 |

### Per-context plain append

| Context | Before control | Before long | After control | After long |
| ------- | -------------: | ----------: | ------------: | ---------: |
| 1       |        28.3 ms |    290.6 ms |        7.4 ms |    14.9 ms |
| 2       |        31.6 ms |    274.1 ms |        7.5 ms |    15.6 ms |
| 3       |        26.5 ms |    285.8 ms |        9.8 ms |     9.5 ms |

### Per-context streaming median

| Context | Before control | Before long | After control | After long |
| ------- | -------------: | ----------: | ------------: | ---------: |
| 1       |        25.0 ms |    275.0 ms |        5.3 ms |     6.5 ms |
| 2       |        26.4 ms |    266.2 ms |        6.1 ms |     6.6 ms |
| 3       |        25.6 ms |    276.6 ms |        7.3 ms |     6.5 ms |

The fixed run passed all issue thresholds and preserved the full highlighted
DOM, the last generated line (`item_0049`), the durable tail, and all five
streaming deltas. See `after.png` for the captured real UI state.
