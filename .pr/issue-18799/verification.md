# Issue #18799: conversation launch verification

Base: `11eba891cb2455f87253888d9d3bf3f6b0f12bc1`.

## Behavior

The home launcher waits for the create mutation before navigating. The mutation
previously awaited an optional installed-plugin metadata query after the Agent
Server had already created the conversation. It now returns the conversation ID
independently of that query and merges the eventual enabled-plugin coordinates
into the original conversation's metadata. The lookup captures the launch backend
and uses the existing backend/org query key.

## Automated checks

Run from the repository root with Node 24.15.0 and npm 11.12.1:

```sh
npm test -- __tests__/api/use-create-conversation-metadata.test.ts \
  src/api/plugins-management-service.test.ts \
  __tests__/hooks/mutation/use-create-conversation.test.tsx
npm test
npm run lint
npm run build
npm run build:lib
```

Recorded results:

| Check | Result |
| --- | --- |
| Focused mutation/service tests | 36 passed |
| Full Linux unit suite | 855 files passed; 9,501 tests passed; 7 todo |
| TypeScript, ESLint, Prettier (`npm run lint`) | Passed |
| Application and library builds | Both passed |

Five new regression cases cover pending lookup completion, eventual coordinate-only
metadata merge, rejected lookup, concurrent launches on different backends, and
explicit backend selection in the plugin service. All five reproduce on the
unmodified production source. The final four changed source/test files were
SHA-256 matched to the files used for the full Linux run.

The two existing `mock-llm-model-switch.spec.ts` cases completed in a normal
serial run together with `mock-llm-profile-management.spec.ts`: initial response,
`/model`, the `switch_llm` POST, post-switch response, and no-error-banner
assertions all ran without changing their timeouts or assertions.

## Running Canvas demonstration

Environment: Canvas 1.33.0, real Agent Server 1.58.1 and automation 1.25.1,
Python 3.12.15, Chromium 153.0.8010.12. Model responses use the repository's
deterministic mock-LLM server. Conversation creation, REST requests and navigation
use the real local stack.

The temporary [Playwright reproducer](reproduce-navigation.spec.ts.txt) forwards
the actual `GET /api/plugins/installed` response unchanged, holding only its
delivery until after the navigation check. It releases the response and deletes
its own conversation during cleanup. To repeat with the project's mock-LLM
Playwright configuration:

```sh
cp .pr/issue-18799/reproduce-navigation.spec.ts.txt \
  tests/e2e/mock-llm/regressions/mock-llm-plugin-metadata-navigation.spec.ts
npm run build:app
npm run test:e2e:mock-llm -- \
  tests/e2e/mock-llm/regressions/mock-llm-plugin-metadata-navigation.spec.ts
```

The recorded comparison uses the same backend pins and real-response hold:

| Version | Observation after the successful create response |
| --- | --- |
| Unmodified base | Still on `/` after 5,307 ms with the plugin response held |
| Fix | Opened `/conversations/<id>` after 535 ms while the response was still held |

The baseline observation ends at the run's overall timeout; it is not a 30-second
post-create measurement. These injected-delay observations demonstrate that
navigation no longer depends on the optional response, rather than measuring
ordinary production latency.

- Before: [screenshot](before-pending.png), [video](before.webm), [timing](before-timing.json).
- After: [screenshot](after-pending.png), [video](after.webm), [timing](after-timing.json).

The after screenshot shows the created conversation open and connecting; it is
evidence of navigation. The original model-switch cases separately exercise
model responses and switching. The reproducer is kept here as review evidence,
outside the permanent suite, so it does not change the existing journey order.
