# History projection repair

Loading an older page must leave one result card per tool call. This fixes #17559 / #16947, a preexisting issue distinct from the performance changes in #19266.

The baseline is `754abee4f0869dfc85dbd0acb6457f2e0ad17d4c`. The tested product revision is `2b8fd1fe95b571ead2ac6a9f3cc45ee7eb3c1d73`. Later commits contain review artifacts only.

## Real application proof

Both isolated stacks used Agent Server/SDK 1.62.0 and automation 1.32.1. Canonical SDK persistence arranged 49 terminal action/result pairs across a 50-event page boundary. No model or command ran. Canvas fetched real REST history and connected its real WebSocket; there was no route interception or browser-store injection.

| Check | Before | After |
| --- | --- | --- |
| Desktop and phone, two attempts each | 50 cards for 49 calls; command 025 twice | 49 cards; command 025 once |
| Separate equal-timestamp page boundary | 100 unique messages, ordered | 100 unique messages, ordered |
| Tool disclosure and unsent draft typing | Pass | Pass |

Desktop's collapsed transcript automatically loaded the older page. Phone started with 25 calls and loaded the older page on scroll. A separate Chrome extension check also observed 25 → 49 calls and exactly one row per command after opening the group. Phone means browser viewport emulation, not a physical device.

[Before desktop](before-desktop.png) · [After desktop](after-desktop.png) · [Before phone](before-phone.png) · [After phone](after-phone.png) · [Before video](before.mp4) · [After video](after.mp4)

[Reproduction recipe](recipe.md) · [Run identities, results and limits](verification.json) · [Before observations](before-observations.json) · [After observations](after-observations.json) · [Sanitized ledger](ledger.jsonl)

Verification skills and maps remained read-only. The map has no executable known-bug block for this issue, so this is a reproducible ordinary Control check, not a `bug proof` verdict. Both captured runs and the unused preliminary run were stopped; all owned ports closed.

## Focused checks

- Seven newly added regression cases failed against baseline source; the existing 12 store tests passed. [Red log](baseline-tests.log)
- 398 distinct tests passed, with one existing todo, across 17 files. These cover the store, helper, Local/Cloud history, pagination, WebSocket providers, shared viewer, transcript export, grouping, disclosure, model messages and render stability. [114 focused](focused-tests.log) · [232 integration](integration-tests.log) · [52 rendering](render-tests.log)
- Independent old/new single-helper and batch parity: 3,000 deterministic comparisons, zero differences. [Receipt](independent-parity.json)
- `npm run lint` passed with zero errors and 140 existing warnings; app and library builds passed. [Lint](lint.log) · [Library build](build-lib.log)
- Configured mock-LLM Playwright regression, `loads older events when scrolling up`: one pass in 15.3 seconds. This existing test intercepts its pagination fixture; it supplements the separate real-backend proof. The first launch safely rejected occupied defaults, then the test ran with isolated ports. [Log](configured-pagination-e2e.log)

## Scaling and scope

[Measured synthetic scaling](scaling.json) compares 1,000 and 10,000 durable events using the actual bundled production functions. Indexed ID reads grew 10× (4,000 → 40,000); the naive sequential fold grew about 99×. Local median store backfill was 0.226 → 1.573 ms. Timings are noisy local function measurements, not browser latency. Run `node .pr/projection-proof/scaling.mjs` from this checkout to repeat the probe.

The batch projection shares replacement rules with the incremental helper. Chronological single-event and token paths retain their existing behavior. Reordered history preserves active slot content, identity and planning provenance; REST continuation and tie ordering remain owned by the event store.

Local and Cloud feed the same store; ACP pairs and planning provenance have focused coverage. Real-app evidence covers Local OpenHands standard conversations. It does not claim live Cloud, ACP execution, planning/delegated lifecycle or embedded-host verification. No event wire contract, server minimum, dependency or migration changes. Raw history and DOM retention remain unbounded; this PR does not implement virtualization.

Prepared by an AI agent on behalf of Engel Nyst.
