## Tracking / Analytics Architecture

One Canvas-owned PostHog client owns telemetry and app analytics.

- `src/services/telemetry.ts` is the only module that accesses the named `agent-canvas` PostHog client. The name isolates Canvas identity, persistence, configuration, and consent from an embedding host's default singleton. React code declares Cloud user identity and event context through the service and captures through the service; it never receives, identifies, or resets the SDK client directly.
- `TelemetryProvider` configures bootstrap/runtime options, eagerly initializes the service, and is the sole owner of the `useTelemetry()` lifecycle that emits install/session events. Do not mount that lifecycle hook separately in Canvas routes or internal components. The provider does not expose PostHog context or maintain a second client lifecycle.
- The default PostHog key and direct ingestion host live in `config/defaults.json` under `telemetry`. Local launchers (`dev-with-automation`, `dev-static`, published binary path) and Docker default `AUTOMATION_POSTHOG_API_KEY` from explicit automation env, then `VITE_POSTHOG_API_KEY`, then that shared default key, so the automation backend can emit local consent-gated telemetry without extra user config. Keep `VITE_DO_NOT_TRACK=1` disabling the zero-config default.
- Unconfigured source builds use the staging key and route through `https://z.openhands.dev`. Release workflows pass the public production key through `VITE_POSTHOG_API_KEY`. Precompiled npm consumers override `apiKey`, `apiHost`, and `uiHost` at runtime through `AgentServerUIProviders.analytics` or `configureTelemetry()`.
- `setTelemetryConsent` is the only user-consent controller; `configureTelemetry(false)` is the embedding host's hard disable. An explicit first-run browser decision remains pending across local backends until `useSyncTelemetryConsent` persists it to Cloud; a stale/default backend value must not overwrite that newer choice during login or navigation. Once Cloud confirms the choice, backend `user_consents_to_analytics` changes are authoritative and mirrored to the client. No other hook or component should call `opt_in_capturing` / `opt_out_capturing` directly.
- `subscribeTelemetryConsent` is the sole React-facing consent store. Hooks that render consent state must use `useSyncExternalStore`; do not mirror consent in component state or gate events outside `telemetry.ts`.
- `canvas_install` fires once, pre-consent, with the client's anonymous distinct ID. After consent and Cloud authentication, Canvas identifies PostHog with the stable Cloud user ID so PostHog joins the earlier anonymous activity to that person. Merely switching to a local backend clears Cloud event context without resetting the identified person; a resolved logout/account change, consent revocation, or privacy clear owns the reset. Local-only and never-authenticated traffic remains on the anonymous browser/install ID. Cloud account context (`cloud_user_id`, `cloud_user_email`, `cloud_org_id`) is attached as event properties only while a Cloud backend is active.
- `telemetry.ts` adds immutable `client_source`, `client_version`, `package_name`, and `package_version` properties in `before_send`, so reset cannot remove attribution and event producers cannot override it. Repeated business milestones use deterministic PostHog `$insert_id` values instead of process-local caches.
- `trackEvent` and `useTelemetry` remain the public library telemetry API for npm consumers (the `TelemetryConsentBanner` component was removed; hosts needing a consent UI build their own on `useTelemetry`). Non-React state machines use typed functions in `cloud-funnel-analytics.ts`; they do not call `trackEvent` directly.
- React app events use typed functions in `src/hooks/use-tracking.ts`; components never call `posthog.capture()` raw. The hook attaches `current_url` automatically and captures through the telemetry service; Cloud account email is attached centrally as `cloud_user_email` while Cloud context is active. It may read backend settings for event properties, but must never gate capture on a settings snapshot: `useSyncTelemetryConsent` has already mirrored the authoritative decision to the telemetry service, and settings can be stale during a backend transition.
- A business milestone has one canonical event capture. Do not conditionally switch between telemetry and app clients or emit duplicate events.

### Cloud funnel observability
- OAuth device authorization and Cloud conversation-start requests include the coarse `X-OpenHands-Client: agent_canvas` and `X-OpenHands-Client-Version` headers from `src/api/client-source.ts`. Never put device codes, API keys, conversation content, raw hosts, or other user data in these headers.
- Production ingress must retain those two headers as structured Datadog facets before source-specific operational queries will work.
- The consented OSS funnel uses typed `cloud_device_authorization_started`, `cloud_device_authorization_succeeded`, and `cloud_conversation_ready` events from `cloud-funnel-analytics.ts`; React emits the canonical `backend_added` event through `useTracking`.

### Adding a new event
1. Add a typed function to `useTracking` in `src/hooks/use-tracking.ts`
2. Add the function to the hook's `return` object
3. Destructure and call it from the component: `const { trackFoo } = useTracking()`

### Event dictionary: onboarding_link_clicked

One stable event for every onboarding link/CTA click. New onboarding links must
reuse this contract (extend the unions in `use-tracking.ts`), never add one-off
events per destination.

Properties (all values controlled enums or booleans — never raw destination
URLs, query params, or link text; `current_url` is the standard app-page common
property, not a destination):
- `link_id` (`OnboardingLinkId`): `configure_llm` | `start_conversation` |
  `schedule_task` | `customize_agent` | `connect_mcp` | `join_slack` |
  `open_docs`
- `destination_type` (`OnboardingLinkDestinationType`): `community` |
  `integration` | `documentation` | `settings` | `conversation` | `automation`
- `surface` (`OnboardingLinkSurface`): `landing_checklist` |
  `onboarding_modal` (reserved; no modal links are instrumented yet)
- `checklist_item` (optional): the owning checklist item's `link_id`; set on
  every `landing_checklist` emission, including `open_docs` clicks
- `step_id` (optional): reserved for future onboarding-modal links
- `is_external` (boolean): whether the destination leaves the app

Instrumented CTAs (sidebar "Getting started" checklist; the row link and its
preview action CTA intentionally share one `link_id` — same destination):

| Checklist item | Row + preview action | Preview docs link |
|---|---|---|
| Add LLM API key | `configure_llm` / `settings` / internal | `open_docs` / `documentation` / external |
| Start your first chat | `start_conversation` / `conversation` / internal | `open_docs` |
| Schedule a task | `schedule_task` / `automation` / internal | `open_docs` |
| Customize your agent | `customize_agent` / `settings` / internal | `open_docs` |
| Connect an MCP integration | `connect_mcp` / `integration` / internal | `open_docs` |
| Join the OpenHands Slack | `join_slack` / `community` / external | `open_docs` |

Excluded CTAs (per the one-canonical-capture rule above):
- Onboarding-modal wizard controls (back/next/skip/close, agent cards) →
  covered by `onboarding_step_viewed` / `onboarding_completed` /
  `onboarding_skipped`
- Modal backend-connect CTAs and the backend form's docs links →
  `backend_added` with `source: "onboarding"`
- LLM settings help links inside the embedded settings screen (shared with
  non-onboarding surfaces) → setup outcome captured by `settings_saved`
- Recommended-automation cards → `prebuilt_automation_enabled`
- Checklist expand/collapse toggle and the settings visibility switch → UI
  state, not destination links

Known limitation: middle-click (`auxclick`) opens are not captured; tracking
uses React `onClick` only and never prevents default navigation.

### Env vars
`VITE_POSTHOG_API_KEY` is the sole build-time PostHog key. Unconfigured source builds use staging; official release workflows set production explicitly. Precompiled consumers use runtime configuration instead.
