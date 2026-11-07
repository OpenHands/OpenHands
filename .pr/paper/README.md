# Paper theme evidence

Paper adds warm paper surfaces, brown ink, rust actions, and local Newsreader
headings to the existing Canvas layout. The name and identifier are `Paper` and
`paper`; Liberty Labs is the visual inspiration. The existing default and saved
selections remain unchanged.

## Capture setup

These are screenshots of the running Agent Canvas mock build in Chrome, using
existing MSW conversation and Git fixtures. They demonstrate frontend rendering,
not a live Agent Server or LLM run. No model was configured or executed.
The LLM-setup notice and reconnecting indicator are expected in this setup.

Desktop captures are 1440 × 1000; narrow captures are 390 × 844. `before-*` is
**the existing Light+ palette on the same build**, not an old-source bug
reproduction. `after-*` is Paper. Fixture conversation names, text, code, and
settings are synthetic. Browser chrome and private machine information are not
included.

| Surface                            | Existing Light+                       | Paper                                        |
| ---------------------------------- | ------------------------------------- | -------------------------------------------- |
| Desktop conversation               | [Light+](before-chat-desktop.png)     | [Paper](after-chat-desktop.png)              |
| Narrow conversation                | [Light+](before-chat-mobile.png)      | [Paper](after-chat-mobile.png)               |
| Desktop settings                   | [Light+](before-settings-desktop.png) | [Paper](after-settings-desktop.png)          |
| Narrow settings                    | [Light+](before-settings-mobile.png)  | [Paper](after-settings-mobile.png)           |
| Narrow selector and keyboard focus | —                                     | [Paper selector](after-selector-mobile.png)  |
| Narrow navigation                  | —                                     | [Paper drawer](after-navigation-mobile.png)  |
| Monaco diff                        | —                                     | [Paper diff](after-diff-desktop.png)         |
| Command menu                       | —                                     | [Paper menu](after-command-menu-desktop.png) |

The terminal's empty state was inspected in Chrome. Populated ANSI output is
covered by the renderer tests, not by these screenshots. Prism and Monaco
mappings, all sixteen terminal ANSI colors, theme restoration, and contrast
have focused coverage.

## Reproduce

1. Install the committed lock with the pinned npm (`npm ci`).
2. Run `npm run dev:mock`, then open Settings → Application → Color Theme.
3. Select Paper. It applies immediately; reload to check persistence.
4. Open the generated-canvas fixture, expand the Commits panel's Uncommitted
   group, and expand a changed file to see the Monaco palette.
5. At 390px width, open the theme selector and navigation drawer using the
   keyboard. Return to OpenHands-Neutral to check restoration.

The committed Playwright suite `tests/e2e/paper-theme.spec.ts` seeds only its
isolated origin and covers these theme-specific flows. For this run, it used a
built mock preview with the shared test runner's browser settings, one worker,
and an isolated local port; the configured shared development port was left
untouched.

## Validation

Asset-location follow-up (`ec644d37f5`): moved the definition, font, stylesheet,
license, and provenance into `src/themes/color-theme/definitions/paper/`.
Both builds and 35 existing packaging/theme tests passed; focused lint and
formatting passed. The emitted standalone font, inlined library font, and both
license copies were verified byte-for-byte against their source files.
The screenshots above predate this source-only move; rendering is unchanged.

The original implementation passed [Ubuntu CI](https://github.com/OpenHands/OpenHands/actions/runs/42289965463)
at `f6c9b33603`: all 887 files and 10,233 tests, lint, both builds, and package
checks; the Windows app build also passed. The local results recorded for that
implementation were:

- `npm run lint`: passed; zero errors, 235 existing warnings; formatting passed.
- `npm run build`, `npm run build:lib`, and `npm run build:mock`: passed. Both
  distribution modes include the full Newsreader OFL license.
- Focused theme runtime and embedding suites: 27 passed.
- Prism, terminal, and Monaco suites: 26 passed.
- Context-window ring and color-theme suites after the hover correction:
  36 passed. Paper's ring track now exceeds 3:1 at rest and on hover.
- Playwright Paper suite: 2 passed in 10.2s; desktop persistence/restoration and narrow keyboard,
  contrast, font loading, responsive layout, and reduced-motion checks.
- Full Vitest run: 883 passing files and 10,216 passing tests; 17 failures across
  four files. One was Paper's hover-ring contrast, fixed and verified above.
  The remaining 16 are in unchanged launcher tests: one occupied fixed test
  port, thirteen local Bash startup/stderr failures, and two local shell child
  failures. The shared listener and shell configuration were left untouched.

No API contract, server compatibility floor, dependency version, telemetry
behavior, or conversation state is changed. Local/Cloud, OpenHands/ACP, and
standard/planning/delegated paths share the existing theme owner. Standalone
bootstrap and embedded wrappers remain separate, covered by the existing
runtime and style-isolation suites.

A fresh independent visual review assessed desktop/narrow captures against the
pinned Liberty Labs direction and existing Canvas structure: **ship**, with no
material visual fixes. It confirmed heading-only serif type, quiet flat paper
surfaces, readable supporting ink, and no visible clipping. The one-time design
detector found only two pre-existing gradient-text rules outside this change.
