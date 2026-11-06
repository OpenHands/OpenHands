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
- `npm test -- --maxWorkers=4` completed: 889 test files passed and two
  unchanged-main Docker shell suites failed; 10,272 tests passed, 15 failed,
  one skipped and seven were marked todo. The full run was not all green.
  All 13 failed KV-secret cases passed when their complete 30-test suite was
  rerun with the real system Bash and `--norc`, preventing this machine's
  personal startup file from leaking into the fixtures. The two session-key
  policy cases still fail because macOS Bash 3.2 treats an empty array under
  `set -u` as unbound; they need the newer Bash used in the Docker environment.
  Those Docker tests and entrypoint are identical to the merged upstream main.
  No test was skipped or weakened to obtain the targeted results.

Localhost-binding checks and app prerendering ran with permission to bind their
own temporary test servers. Initial sandbox-denied attempts were rerun; no
shared service was restarted. Source checks alone do not establish live Voice,
microphone, speaker, iPad or browser-flow evidence. The integrated pass below
records the browser coverage and its limits separately.

## Integrated browser verification

The isolated pass used Canvas `0eedf952e14373a2b0a613b9333daa9b47b64009`
and Agent Server `f3ae7b564edd2eac7aa8902d601c4da699ac4209`. It began with
App `9175d281fef8f94517bc3c49e9e0df9eb094284c`, then used the actual Apps UI to
update to `f97b109a144725d7a9ce3a3f11d2eeb991c5037e`. The checks below used
synthetic conversations on that isolated backend, with a real Luna subscription
profile. Screenshots and the detailed ledger are retained privately.

- The saved OpenHands Cat used its terminal to calculate `2 + 3` and returned
  `5`. Opening the regular chat and sending a follow-up continued the same Cat.
- Normal Open chat → sidebar Projects navigation retained an unsent App draft
  and the same history. The idle companion remained visible with Talk and Open
  chat controls outside Projects. An initial draft-loss observation came from
  the verification CLI's full browser navigation; it was not reproduced with
  normal in-app navigation and is not a product regression.
- Regular-view `/new` opened the new-Cat flow without eagerly creating a
  conversation. On the final App bundle, sending created a distinct second
  Cat, and selecting the first resumed its saved history.
- App `/condense` retained the Cat identity. Its saved event count went from
  18 to 21: exactly the condensation request, result and state update. All 18
  previous events were unchanged; no extra agent action was recorded.
- After that compaction on the final App bundle, saved events showed
  `invoke_skill(openhands-api)` followed by a terminal query of the advertised
  isolated backend. The answer was two conversations, matching the two
  synthetic Cats independently present on that backend.
- At a 390-pixel viewport, Projects and the regular conversation panel both
  had a document width of 390 pixels, with a visible Cat badge in the panel.
  At 820 pixels, the document width was also 820. Reduced motion showed the
  resting Cat state. These are browser viewport checks, not a physical iPad
  result.
- Voice preflight reported a missing OpenAI key before capture, and no peer
  connection or microphone capture started. This proves that availability
  failure is surfaced without opening media. It does not prove a fresh live
  Voice call, spoken delegation, microphone or speaker playback.

This pass did not exercise Docker-backed worker selection or a physical audio
round trip. Its local-runtime results do not replace those remaining checks.
