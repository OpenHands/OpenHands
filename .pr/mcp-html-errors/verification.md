# MCP error display verification

Issue: [#18862](https://github.com/OpenHands/OpenHands/issues/18862).

Baseline: upstream main `b7fdc6c312039a0b840fdcd34a5c23ca7e65c218`.
Product fix: `42a6d74577c3836868b390ab6c53be421313e85e`.
Verified on 2026-10-25 with production builds and the repository's unchanged
`verify-openhands` Playwright driver. Both isolated stacks passed `doctor`:
Agent Server 1.58.1, automation 1.25.1, Chromium 156.0.8078.47 on Debian 12,
Node 24.21.0 and npm 11.12.1. No model calls or mocked app/backend responses
were used. HubSpot requests reached the real provider.

## HubSpot install failure

At desktop (1440 x 1000) and phone (390 x 844), open `/mcp`, select HubSpot in
the Library, and press Install. Wait for the error. Cancel, reload, and verify
that the Installed section remains empty. Repeat once more at each viewport.

| Observation                | Baseline                                      | Fixed                                                |
| -------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| Error text length          | 9,704 characters, including the HTML document | 110 characters, ending in `Registration failed: 404` |
| Desktop error width        | 470 px box, 65,038 px scroll width            | 470 px box and scroll width                          |
| Phone error width          | 301 px box, 65,038 px scroll width            | 301 px box and scroll width                          |
| Desktop Install button top | 1,200 px, below the viewport                  | 652 px, fully visible                                |
| Phone Install button top   | 1,680 px, below the viewport                  | 620 px, fully visible                                |

All eight attempts gave the same result for their build/viewport. After the
fix, both Install and Cancel are visible and usable, the dialog fits the
viewport, the error wraps, and nothing is installed by the failed attempt.
HubSpot's registration failure still occurs; this change fixes its display.

| Viewport           | Before                                   | After                                  |
| ------------------ | ---------------------------------------- | -------------------------------------- |
| Desktop recording  | [before-desktop.gif](before-desktop.gif) | [after-desktop.gif](after-desktop.gif) |
| Phone recording    | [before-phone.gif](before-phone.gif)     | [after-phone.gif](after-phone.gif)     |
| Desktop screenshot | [before-desktop.png](before-desktop.png) | [after-desktop.png](after-desktop.png) |
| Phone screenshot   | [before-phone.png](before-phone.png)     | [after-phone.png](after-phone.png)     |

## Long plain-text error

Through Add custom server, select STDIO, enter the name `qa-wrap-18862`, and use
the nonexistent command `qa-missing-` followed by 220 `x` characters. Press
Test connection and scroll to its result. The real backend reports that the
command could not start. This was tested on both builds at both viewports.

The entire error, including the 231-character command, is preserved. On the
baseline the paragraph itself grows to 1,827 px, wider than its form. With the
fix it wraps within 470 px on desktop and 301 px on the phone. The phone form
still scrolls vertically to accommodate its fields and error.

| Viewport | Before                                               | After                                              |
| -------- | ---------------------------------------------------- | -------------------------------------------------- |
| Desktop  | [before-plain-desktop.png](before-plain-desktop.png) | [after-plain-desktop.png](after-plain-desktop.png) |
| Phone    | [before-plain-phone.png](before-plain-phone.png)     | [after-plain-phone.png](after-plain-phone.png)     |

All screenshots and sampled frames from all four recordings were visually
inspected. The recordings have no cut waits, missed captures or duration caps.
All browser workflows reported zero page/app errors and no page horizontal
overflow. Baseline run: `2026-10-25T1218-50b05d`; fixed run:
`2026-10-25T1228-14595c`. Both stacks were stopped after verification.

## Regression and quality checks

Before changing production code, the new HTML cases fail six assertions in
`mcp-test-error-message.test.ts`; its other eight cases pass. With the fix:

```sh
npm test -- __tests__/utils/mcp-test-error-message.test.ts __tests__/components/features/mcp-page __tests__/components/features/settings/mcp-settings --maxWorkers=2
```

All 103 tests across seven files pass. The new cases cover document markers,
case variations, each error kind that displays provider details, and verbatim
preservation of plain text and non-HTML angle-bracket content. The shared
formatter serves the install modal, custom editor, and server health label;
the existing component tests for those consumers also pass.

Final `npm run lint` (including TypeScript, ESLint and formatting) passes with
508 existing warnings and zero errors. Changed-file Prettier and
`git diff --check` pass. `npm run build:app` (the command behind
`npm run build`) and `npm run build:lib` pass.

The full Linux suite also passes: 856 files, 9,526 tests and seven existing
todos (665.84 seconds, exit 0). The command was
`env -u DO_NOT_TRACK -u VITE_DO_NOT_TRACK npm test -- --maxWorkers=4 --reporter=dot`.
The verification image's telemetry overrides were unset only for this test
process so the telemetry tests could exercise their normal defaults.
