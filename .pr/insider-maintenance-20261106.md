# Insider Canvas maintenance — 2026-11-06

## Source refresh

Merged OpenHands/OpenHands main `8a0cd446ce2fd211a029c453e32f34c04ff1a859`
into the existing Insider PR head `b6ff25af6e37e99d65aa368225fe4fd4c7b46e36`
without rewriting its history.

The merge retains the current extension-page title/runtime behavior, normal
conversation launch flow and reorganized conversation panel. The extracted
panel row now carries the parent conversation ID to both its card and preview,
so a tagged child cannot be mistaken for the top-level Cat. A panel regression
exercises that distinction through the real card rendering.

The shared compaction tests exercise the real mutation and its single-error
handling alongside the Insider context-change notification and disabled-state
guards. Transcript tests now use the active backend identity required by main's
query scope validation. Vite proxy tests restore the shared network interceptor
before the newer MSW teardown.

Main now owns feature family F29 for ACP sessions. The current Insider family
is F30, with 22 sub-features; the complete map has 30 families and 858 IDs.
Dated October evidence keeps its original F29 identifiers. The current legacy
identity contract accepts top-level Cats with an absent role while excluding
children and explicit non-controller roles.

## Local checks

- Exact locked dependencies installed with `npm ci`.
- Targeted Insider, compaction, extension-runtime, transcript, panel and Vite
  regression suites passed. Verification harness/media/rebuild suites passed.
- `npm run lint` passed, including typecheck and source formatting, with 242
  nonfatal style warnings.
- `npm run build` and `npm run build:lib` passed.
- `npm run check-translation-completeness` passed.
- `control-openhands map check` passed: 30 files, 858 IDs, no problems.
- `git diff --check` passed.
- The broader `npm test -- --maxWorkers=4` run is in progress; its final result
  will be recorded before this maintenance pass finishes.

Localhost-binding checks and app prerendering ran with permission to bind their
own temporary test servers. Initial sandbox-denied attempts were rerun; no
shared service was restarted. These source checks do not establish fresh live
Voice, microphone, speaker, iPad or browser-flow evidence. Integrated UI results
are recorded separately below when available.
