# PR: fix(frontend): make usage-panel cost transparent and attributable

## Problem

The Usage panel's **Total Cost** is the agent-server's `accumulated_cost` — an estimate from litellm's static pricing table, recorded only for calls that could actually be priced. Nothing in the UI says that. A session with 20.1M tokens next to `$0.0030` reads as a billing figure and as "this session was free". Both wrong: the provider billed per its own rate card, and the tiny (or zero) number usually means *pricing data was missing*, not *no spend*.

Server-side cost pipeline (unchanged here, documented in the linked issue):

```
usage id (agent | condenser | profile:<name> | …)
  └─ per-call cost = litellm model_cost[resolved model]
                      (+ proxy cost header / custom_rate override)
     └─ MetricsSnapshot.accumulated_cost   (0.0 when model unpriceable)
        └─ WS ConversationStateUpdateAgentStats / conversation stats REST
           └─ UI combineUsageMetrics() sums ids → "Total Cost"
```

The per-usage-id detail travels all the way to the UI and is then dropped: the panel sums it away. That is the one piece of data that makes the total *attributable*.

## What changes

| Where | Change |
|---|---|
| Usage tab total | estimate note: *"Estimated from model pricing — may differ from what your provider bills."* |
| Zero-cost case | tokens > 0 but cost == 0 → *"No cost data reported for the models used — pricing may be unavailable."* instead of a bare `$0.0000` |
| New "By model" section | per-usage-id rows (model name, its accumulated cost, its input/output tokens) under the total, rendered when more than one usage id contributed |
| Data flow | `usage_breakdown` threaded through `agent-server-conversation-service.types.ts` → `combineUsageMetrics` (keys `usage_to_metrics` by usage id) → WS stats handler → metrics store → `useLiveConversationMetrics` |
| i18n | 3 new keys (`CONVERSATION$COST_ESTIMATE_NOTE`, `CONVERSATION$NO_COST_DATA`, `CONVERSATION$BY_MODEL`) translated for all 16 locales; `declaration.ts` regenerated via `make-i18n` |
| ACP plan note | existing plan-usage note no longer hides the cost section |

The combined **Total Cost** value itself is unchanged — same `accumulated_cost` sum as before. What changes is what the panel tells you about that number.

## Before / after

Preview: (rendered mock: [htmlpreview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/seberatolmez/OpenHands/fix/usage-panel-cost-transparency/.pr/design.html))

```
Before                                  After
──────────────────────────────         ──────────────────────────────
Token Usage                             Token Usage
Input      19,747,245                   Input      19,747,245
Cache Hit  18,989,568                   Cache Hit  18,989,568
Cache Write        0                     Cache Write        0
Output      415,535                      Output      415,535
Total      20,162,780                   Total      20,162,780
Total Cost    $0.0030                   Total Cost    $0.0030
                                        ⚠ No cost data reported for the
                                          models used — pricing may be
                                          unavailable.
                                        By model
                                        openai/gpt-4o-mini   $0.0030
```

## Files

```
src/api/conversation-service/agent-server-conversation-service.types.ts  | UsageBreakdownEntry type + MetricsSnapshot field
src/utils/conversation-metrics.ts                                        | combineUsageMetrics aggregates usage_to_metrics
src/contexts/conversation-websocket-context.tsx                          | stats handler maps breakdown into store
src/stores/metrics-store.ts                                              | breakdown state + reset
src/hooks/use-live-conversation-metrics.ts                               | exposes breakdown (also: per-field selectors)
src/components/features/conversation/usage-panel/usage-panel.tsx         | estimate note + zero-cost flag + ACP note placement
src/components/features/conversation/usage-panel/model-breakdown.tsx     | new "By model" section
src/components/features/conversation/metrics-modal/cost-section.tsx      | estimate note + no-cost-data state
src/i18n/translation.json (+ declaration.ts via make-i18n)               | 3 keys × 16 locales
```

## Testing

- **New**: `__tests__/utils/conversation-metrics.test.ts` (breakdown aggregation, ordering, backward-compat with absent `usage_to_metrics`), `__tests__/components/features/conversation/usage-panel/model-breakdown.test.tsx`, `__tests__/components/features/conversation/metrics-modal/cost-section.test.tsx` (estimate note + no-cost-data state)
- **Updated**: agent-server-adapter (#16480 test now also asserts the breakdown), use-live-conversation-metrics, and the two context-compaction hook test fixtures
- Full suite: **816 files / 9,297 tests passing**; `npm run typecheck`, `eslint src`, `prettier --check`, and translation-completeness all clean

## Human review checklist

- [ ] Wording of both notes (EN strings; locales auto-checked for completeness, not fluency)
- [ ] "By model" visibility rule: shown only when ≥ 2 usage ids contributed — render for `n == 1` too?
- [ ] Zero-cost flag threshold: triggers when accumulated cost is exactly 0 with tokens > 0
- [ ] Backend contract: relies on `usage_to_metrics[*].model_name` staying in the stats payload

## AGENT section

- Agent: OpenHands
- This PR was produced end-to-end by an OpenHands agent at the user's request; the user reviewed the analysis and is submitting it.
