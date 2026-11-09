# ACP command focus border — PR 17730

Review: https://github.com/OpenHands/OpenHands/pull/17730#discussion_r4296907641

The ACP command textarea's resting dark border overrode its focus border in the
compiled Tailwind cascade. The fix adds the missing `dark:focus:border-contrast`
without changing the earlier shared-control focus fix.

## Real-app evidence

Captured from this checkout's production frontend on an isolated local stack at
`http://127.0.0.1:32770`, using real Agent Server **1.56.0**, Automation, ingress,
and Chromium. No MSW or mocked backend was used for these captures. No model was
invoked: the changed behavior is the settings control's CSS/keyboard focus.
The verification CLI's `doctor` passed, including authenticated settings access,
backend version, served-build identity, and absence of page errors. Before and
after runs used the same pinned backend and launch settings, each with isolated
state. Existing user services and state were untouched.

Reproduce: Settings → Agent → Add agent profile → Agent → ACP (external
subprocess). Scroll to Command. Click the textarea, press Shift+Tab then Tab,
and inspect its border while it has keyboard focus. Choose Light+ or
OpenHands-Neutral under Settings → Application → Color Theme.

| Theme / state            | Before                           | After                |
| ------------------------ | -------------------------------- | -------------------- |
| Neutral resting          | `rgb(113, 120, 136)`             | `rgb(113, 120, 136)` |
| Neutral keyboard focused | `rgb(113, 120, 136)` (no change) | `rgb(255, 255, 255)` |
| Light+ keyboard focused  | `rgb(31, 31, 31)`                | `rgb(31, 31, 31)`    |

- [Before, dark resting](before-dark-resting.png)
- [Before, dark focused](before-dark-focused.png)
- [After, dark resting](after-dark-resting.png)
- [After, dark focused](after-dark-focused.png)
- [Before, Light+ focused](before-light-focused.png)
- [After, Light+ resting](after-light-resting.png)
- [After, Light+ focused](after-light-focused.png)
- [Existing shared field, dark](after-dark-existing-field.png)
- [Existing shared field, Light+](after-light-existing-field.png)

## Validation

- Browser regression first failed on the real rendered ACP textarea: expected
  focused white, received the resting gray. It uses compiled CSS and theme
  variables, not class-string assertions.
- Browser coverage exercises Neutral, Deepsea, Neo, Light+, and Solarized Light
  in Chromium, Firefox, and WebKit, including the existing shared fields.
- All 120 theme tests pass after reconciling current main's semantic feedback,
  link, and focus contrast requirements with the PR's light surfaces.
- Focused unit suite: **237 passed** across 12 files (themes, ACP settings,
  shared form controls, hook messages, context meter, run-phase date handling),
  using Node 24 and `TZ=UTC`.
- `npm run lint`: passed (0 errors; existing warnings).
- `npm run build` and `npm run build:lib`: passed.
- Complete unit run: 9,344 passed, 7 todo, 5 failures. Two theme failures were
  fixed and rerun successfully. The date assertion passed with CI's UTC
  timezone (the macOS host prints 1969 for midnight UTC Jan 1 1970). The remaining
  two Docker-policy tests are unchanged from main and hit macOS Bash 3.2's
  empty-array/unbound-variable behavior. No Docker-policy product change was made.

## Merge resolution

Merged upstream main `d01d491d1bd03a37af624bfdc681df813197822e` normally, retaining
both histories. Kept main's semantic feedback/link migrations and Escape-boundary
handling, the PR's portal theme scope, stronger light borders and opaque status
fills. Retained main's darker link tokens and reconciled light focus/error tokens
to pass both branches' contrast contracts. No force push, PR merge, review
dismissal, or comments are part of this update.
