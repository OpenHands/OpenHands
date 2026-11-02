# Template filter scroll verification

Issue: [#18858](https://github.com/OpenHands/OpenHands/issues/18858).

The baseline is upstream main `b7fdc6c312039a0b840fdcd34a5c23ca7e65c218`.
The product fix is `d3903718dd5c5c28ef664e7dd3e2422ebfe529e4`.
Captures were made on 2026-10-25 with the repository's unchanged
`verify-openhands` CLI and its Playwright browser driver, using production app
builds, Agent Server 1.58.1 and automation 1.25.1. Both runs passed `doctor`.
The browser was Chromium 156.0.8078.47 on Debian 12; Node was 24.21.0 and npm
11.12.1. No model calls or mocked app/backend responses were used.

## Browser workflow

For each build, at desktop (1440 x 1000) and phone (390 x 844):

1. Open Automate, then Templates. Confirm all 23 cards are present.
2. Scroll the filter rail to its right end and select Tavily (one matching card).
3. Scroll to the right end again and select Event (no matching cards).
4. Scroll to the right end again and deselect Event (the matching card returns).
5. Search for `qa-no-matching-template-18858`, then use Clear filters. Confirm
   the search and selections clear and all 23 cards return.

The explicit scroll before toggling Event keeps the starting position identical
on both builds; on a phone, Playwright may otherwise scroll the partially
clipped Tavily chip into view when selecting it.

Each workflow was repeated twice per viewport per build. The observations were
the same on both attempts:

| Transition | Baseline desktop | Fixed desktop | Baseline phone | Fixed phone |
| --- | --- | --- | --- | --- |
| Enter empty results | 407 -> 0 px | 407 -> 407 px | 835 -> 0 px | 835 -> 835 px |
| Return to a matching card | 407 -> 0 px | 407 -> 407 px | 835 -> 0 px | 835 -> 835 px |

These are the filter container's measured `scrollLeft` values. All eight
attempts restored the catalog with Clear filters and had no page horizontal
overflow, browser page errors or app console errors. Screenshots and sampled
frames from every recording were visually inspected. The phone recording
shows the selected Event chip remaining visible; the rail still requires
horizontal scrolling to see its entire contents.

## Captures

| Viewport | Before | After |
| --- | --- | --- |
| Desktop recording | [before-desktop.gif](before-desktop.gif) | [after-desktop.gif](after-desktop.gif) |
| Phone recording | [before-phone.gif](before-phone.gif) | [after-phone.gif](after-phone.gif) |
| Desktop empty-state screenshot | [before-desktop.png](before-desktop.png) | [after-desktop.png](after-desktop.png) |
| Phone empty-state screenshot | [before-phone.png](before-phone.png) | [after-phone.png](after-phone.png) |

The recordings show the actual transitions and Clear filters. They contain no
paused or cut waits. Baseline run: `2026-10-25T1202-018f60`; fixed run:
`2026-10-25T1211-67aa60`. Both isolated stacks were stopped after verification.
This is a manual CLI verification recipe, not a formal `bug proof` result:
the existing read-only feature map has no executable F22 bug block.

## Regression and quality checks

The new behavioral test fails against baseline production code when the filter
DOM node is replaced. The other 75 tests in the two selected files pass. With
the fix, all 76 tests pass:

```sh
npm test -- __tests__/components/automations/recommended-automations.test.tsx __tests__/routes/automations-dashboard.test.tsx --maxWorkers=2
```

`npm run lint` passes, including TypeScript, ESLint and source formatting
(508 existing warnings, zero errors). The changed test also passes Prettier.
`npm run build:app` (the command behind `npm run build`), `npm run build:lib`,
and `git diff --check` pass.

The complete Linux suite passes: 856 files, 9,519 tests, seven existing todos
(548 seconds):

```sh
env -u DO_NOT_TRACK -u VITE_DO_NOT_TRACK npm test -- --maxWorkers=4
```

The test process clears the verification container's telemetry overrides so
the consent tests can exercise their mocked opt-in/opt-out behavior. The live
app runs kept telemetry disabled.

The native Windows full suite returned 9,449 passes, 34 failures, 36 skips and
7 todos. Rerunning all 13 failing files on unchanged upstream main reproduced
the same 34 failures (135 passes, one skip). They include Unix shell/path/signal
assumptions and a Windows library-output-path assertion. No Windows full-suite
pass is claimed.
