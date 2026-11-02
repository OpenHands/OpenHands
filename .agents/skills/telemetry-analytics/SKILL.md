---
name: telemetry-analytics
description: This skill should be used when the user asks to "add tracking", "add a PostHog event", "change telemetry consent", "instrument onboarding", "debug analytics", or changes telemetry.ts, use-tracking.ts, cloud funnel analytics, or analytics environment variables.
---

# Telemetry and Analytics

Preserve Canvas's single-client telemetry architecture, consent ownership, identity lifecycle, and canonical event contracts.

## Workflow

1. Read `references/guide.md` before changing telemetry, analytics consent, PostHog configuration, Cloud funnel observability, or onboarding event capture.
2. Identify the single canonical owner and capture point for the behavior.
3. Extend typed event functions and controlled property unions rather than calling the SDK directly.
4. Verify consent, identity, backend transitions, and duplicate-capture behavior relevant to the change.
5. Keep secrets, conversation content, raw hosts, and unbounded user data out of telemetry.

## Reference

- **`references/guide.md`** — Telemetry architecture, consent and identity invariants, event creation workflow, Cloud funnel headers, onboarding event dictionary, and analytics environment variables.
