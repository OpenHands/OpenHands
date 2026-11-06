### Description

The Usage panel's **Total Cost** is not what the provider billed for the session, and nothing in the UI says so. With token counts in the millions next to a cost like `$0.0030`, the figure reads as a billing value and as "this session was free" — both wrong.

**Example from a real session:**

```
Token Usage
Input          19,747,245
Cache Hit      18,989,568
Cache Write            0
Output          415,535
Total          20,162,780
Total Cost        $0.0030
```

### How costs are actually calculated today

- The agent-server accumulates cost per usage id (`agent`, `condenser`, `profile:<name>`, …) in `conversation_stats.MetricsSnapshot` (Python-side: `openhands-sdk`'s LLM telemetry, `openhands/llm/utils/metrics.py`).
- Cost comes from a **static lookup** — litellm's `model_cost` map on the *resolved* model string — with optional proxy `cost` response headers and per-call `custom_rate_for_cost_calc` overrides.
- Calls that cannot be priced (unknown or renamed model) are silently recorded as `cost = 0.0` but their tokens still count.
- The frontend just sums those per-usage-id costs (`combineUsageMetrics`) and renders the result as "Total Cost".

So "Total Cost" is: *an estimate of what OpenHands' pricing table says the resolved models cost, for the calls that could be priced — not what any provider billed, and not what a custom/proxied deployment actually pays.*

### Why it is inconsistent with token usage

- Cache reads/writes and per-model price differences are baked into the server-side number; raw token counts in the same card can't be reconciled with it.
- Unpriceable calls contribute 0 cost while their tokens inflate the token totals — a session with `0.0` cost can still have burned 20M tokens (exactly the case above, where an unknown model priced to 0.0).
- The number shown is never refreshed against the provider.

### Steps to reproduce

1. Run a session against a model whose pricing isn't in litellm's `model_cost` (or via a custom provider endpoint).
2. Open the Usage panel: millions of tokens, `Total Cost` ≈ $0.00 (or a small number that doesn't match the provider bill).

### Expected behavior

The panel should not present an opaque number that looks like a bill. It should:
- label the cost as an estimate ("Estimated from model pricing — may differ from what your provider bills"), and
- when tokens were recorded but the accumulated cost is 0, say that pricing data is missing instead of showing `$0.0000`,
- ideally show the per-usage-id (per model) cost breakdown the backend already reports, so the total is attributable.

### Related issues/PRs

- openhands/software-agent-sdk#4382 — inconsistent cost data across providers (related, provider-facing side)
- openhands/software-agent-sdk#4816 / #4817 — cost always 0 (report + fix; covers the "silent 0.0" behavior)
- #19090 — model name `provider/model` missing from token usage details (same display gap, no per-model cost shown)
- #16835 — token usage counting & billing (costs don't align with what cloud charges)
- #17691, #16308 — token/cost display consistency reports

### Suggested change (frontend, minimal)

- estimate note under the combined total in the Usage tab / metrics modal
- `No cost data reported for the models used — pricing may be unavailable.` when tokens > 0 but accumulated cost == 0
- a "By model" breakdown section fed by `usage_to_metrics` (already available; the UI currently drops it)

We have a working implementation of the frontend part on a local branch (`fix/usage-panel-cost-transparency`, 17 files, +589/−7, all tests green, i18n complete for all 16 locales) and will open a PR referencing this issue.
