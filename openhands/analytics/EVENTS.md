# PostHog Analytics — Event Catalog

*Last updated: 2026-05-01*

## Architecture Overview

Analytics is split into three lanes:

1. **Server-side business events** (SaaS only) — captured through a centralized `AnalyticsService`. Covers the core product lifecycle: signups, logins, conversations, credits, activation, onboarding, settings, and team management.
2. **Client-side UI events** (SaaS only) — captured via `useClientAnalytics` hook for UI-only interactions that have no natural server round-trip (e.g. enterprise CTA clicks, lead form submissions).
3. **Frontend automatic instrumentation** (SaaS and OSS) — web vitals, error tracking, network timing, pageviews. No explicit event code required.

Every event respects user consent.

---

## Backend Events

| # | Event | When It Fires | Key Properties |
|---|---|---|---|
| 1 | **user signed up** | New user completes OAuth registration (once per user) | `idp`, `email_domain`, `invitation_source` |
| 2 | **user logged in** | Every successful authentication (Keycloak or device auth) | `idp` |
| 3 | **conversation created** | A new conversation is initialized | `conversation_id`, `trigger`, `llm_model`, `agent_type`, `has_repository` |
| 4 | **conversation finished** | Conversation reaches a successful or stopped terminal state | `conversation_id`, `terminal_state`, `turn_count`, `accumulated_cost_usd`, `prompt_tokens`, `completion_tokens`, `llm_model`, `trigger` |
| 5 | **conversation errored** | Conversation reaches an error or stuck state | `conversation_id`, `error_type`\*, `error_message`, `llm_model`, `turn_count`, `terminal_state` |
| 6 | **conversation deleted** | User deletes a conversation | `conversation_id` |
| 7 | **credit purchased** | Stripe checkout completes successfully | `amount_usd`, `credit_balance_before`, `credit_balance_after` |
| 8 | **credit limit reached** | Conversation fails due to insufficient credits (fires alongside #5) | `conversation_id`, `credit_balance`, `llm_model` |
| 9 | **git provider connected** | User connects a git provider (GitHub, GitLab, etc.) | `provider_type` |
| 10 | **onboarding completed** | User submits the onboarding form | (form selections passed as properties) |
| 11 | **settings saved** | User saves their settings | `settings_changed`\*\* |
| 12 | **trajectory downloaded** | User downloads a conversation trajectory | `conversation_id` |
| 13 | **team members invited** | User invites team members to their organization | `invited_count`, `successful_count`, `failed_count`, `role` |

\*Error types: `budget_exceeded`, `model_error`, `runtime_error`, `timeout`, `user_cancelled`, `unknown`

\*\*`settings_changed` is a list of payload keys that were modified (e.g., `['llm_model']`, `['agent_settings_diff']`, etc.)

Every event also carries: `app_mode` (saas/oss), `is_feature_env`, and `org_id` when available.

**Note:** Backend events fire in SaaS only. The `AnalyticsService` is never initialized in OSS — `get_analytics_service()` returns `None` and all call sites are guarded.

---

## Identity & Group Tracking (SaaS only)

| Action | When | What's Set |
|---|---|---|
| **Identify user** | Login (Keycloak or device auth) | Person: `email`, `org_id`, `org_name`, `idp`, `last_login_at`. Group (org): `org_name`, `member_count`. |
| **Update person** | Signup, org switch | `signed_up_at` on signup; `org_id`, `org_name` on org switch |
| **Update org group** | Login, onboarding | `member_count`, `onboarding_completed_at` |

---

## Client-Side UI Events (SaaS only)

A small set of explicit frontend events captured via the `useClientAnalytics` hook. These are UI interactions with no natural server round-trip — they fire directly through the PostHog JS SDK.

| # | Event | When It Fires | Key Properties |
|---|---|---|---|
| 1 | **enterprise cta clicked** | User clicks "Learn More" on an enterprise CTA (login page, homepage, context menu, device verify) | `location` |
| 2 | **enterprise lead form submitted** | User submits the enterprise contact form | `request_type`, `name`, `company`, `email`, `message` |

---

## Frontend Automatic Instrumentation (SaaS and OSS)

The frontend initializes PostHog in both SaaS and OSS deployments (OSS uses a hardcoded fallback project key).

| Feature | What It Captures |
|---|---|
| **Web Vitals** | LCP, FCP, INP, CLS |
| **Network Timing** | API request latencies |
| **Error Tracking** | Uncaught JavaScript exceptions |
| **Pageviews** | Automatic page navigation tracking |
| **Session Linking** | Correlates frontend sessions with backend events via `X-POSTHOG-SESSION-ID` tracing header (SaaS only) |

Person profiles are created for identified users only. Session replay is **not configured in code** — whether it is active depends on the PostHog project's server-side settings.

---

## Event Lifecycle

```
User signs up       →  user signed up  +  identify
User logs in        →  user logged in  +  identify
Onboarding          →  onboarding completed
Git connect         →  git provider connected
Settings change     →  settings saved

Conversation starts →  conversation created
  ├─ Finishes OK    →  conversation finished
  ├─ Errors         →  conversation errored   (+ credit limit reached if budget)
  ├─ Stopped        →  conversation finished
  └─ Deleted        →  conversation deleted

Credit purchase     →  credit purchased
Team invite         →  team members invited
Trajectory export   →  trajectory downloaded
Org switch          →  person properties updated (no event)
```

---

## Dashboards (Staging Project)

All dashboards below are tagged `analytics-overhaul` in the **Staging** PostHog project (ID 163845). They were created on 2026-03-05/06.

### [Conversion Funnel](https://us.posthog.com/project/163845/dashboard/1334830)

4-step ordered funnel with 30-day conversion window.

| Step | Event |
|---|---|
| 1 | user signed up |
| 2 | conversation created |
| 3 | conversation finished |
| 4 | credit purchased |

### [User Retention](https://us.posthog.com/project/163845/dashboard/1334831)

Weekly trends comparing new signups to returning users who create conversations. Note: this is a trends approximation (signups vs conversation DAU), not a true cohort retention chart.

| Insight | Type | Events |
|---|---|---|
| Weekly Retention: Signup to Conversation | Trends (weekly) | `user signed up` (total), `conversation created` (DAU) |

### [Credit Usage](https://us.posthog.com/project/163845/dashboard/1334832)

| Insight | Type | Breakdown |
|---|---|---|
| Credit Purchased by Org | Trends (weekly) | `credit purchased` by `org_id` |
| Credit Limit Reached by Org | Trends (weekly) | `credit limit reached` by `org_id` |
| Avg Credit Balance After Purchase | Trends (weekly) | avg `credit_balance_after` on `credit purchased` |

### [Churn Signals](https://us.posthog.com/project/163845/dashboard/1334833)

| Insight | Type | Description |
|---|---|---|
| Churn Signal: Credit Limit Without Purchase | HogQL table | Users who hit credit limit in last 90 days with no subsequent purchase |

### [Usage Patterns](https://us.posthog.com/project/163845/dashboard/1334834)

| Insight | Type | Breakdown |
|---|---|---|
| Conversations by Model | Trends (weekly) | `conversation finished` by `llm_model` |
| Conversations by Trigger | Trends (weekly) | `conversation finished` by `trigger` |
| Avg Cost per Conversation | Trends (weekly) | avg `accumulated_cost_usd` on `conversation finished` |

### [Product Quality](https://us.posthog.com/project/163845/dashboard/1334835)

| Insight | Type | Breakdown |
|---|---|---|
| Success Rate by Terminal State | Trends (weekly) | `conversation finished` by `terminal_state` |
| Error Rate by Model | Trends (weekly) | `conversation errored` by `llm_model` |

### [Frontend Health](https://us.posthog.com/project/163845/dashboard/1337892)

| Insight | Type | Description |
|---|---|---|
| Web Vitals -- LCP | Trends (daily, 30d) | Avg Largest Contentful Paint |
| Web Vitals -- FCP | Trends (daily, 30d) | Avg First Contentful Paint |
| Web Vitals -- INP | Trends (daily, 30d) | Avg Interaction to Next Paint |
| Web Vitals -- CLS | Trends (daily, 30d) | Avg Cumulative Layout Shift |
| JS Error Rate | Trends (daily, 30d) | Total `$exception` events per day |
| Top JS Errors | Table (30d) | `$exception` broken down by `$exception_type` |

---

## Consent & Privacy

- All backend events are gated on `user_consents_to_analytics`. No server-side data is sent when consent is absent.
- Frontend consent is synced: `posthog.opt_in_capturing()` / `posthog.opt_out_capturing()` mirrors the backend setting.
- OSS deployments send frontend-only automatic instrumentation (web vitals, errors, pageviews) to a shared PostHog project. No backend business events are sent.
- Feature/staging environments are isolated — distinct IDs are prefixed with `FEATURE_` so test traffic never pollutes production data.
