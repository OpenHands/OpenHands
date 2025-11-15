# V1 API Redesign – Working Notes (tracking)

Purpose: Track current V1 implementation status for the three proposal aspects and enumerate V1 REST routes. Focus on app-server (V1) that orchestrates remote agent-server from agent-sdk. Ignore V0 legacy where not on the V1 execution path.

Sources read in repo:
- openhands/server/app.py (router mounting)
- openhands/app_server/v1_router.py (V1 API surface)
- app-server packages: app_conversation, event, event_callback, sandbox, user, services, config
- Frontend calls under frontend/src/api to validate route usage

External: agent-sdk (agent-server + sdk) is the V1 core. In this tree we import:
- openhands.agent_server.* (models, utils)
- openhands.sdk.* (LLM, workspace, secrets)

---

## 1) UserContext

Status
- UserContext abstraction exists and is used via FastAPI DI.
  - Interface: openhands/app_server/user/user_context.py
    - get_user_id(), get_user_info(), get_authenticated_git_url(), get_latest_token(provider_type), get_secrets()
  - Implementations:
    - AuthUserContext (user-auth backed): openhands/app_server/user/auth_user_context.py
      - Bridges to legacy user auth storage to obtain settings and provider tokens as needed, but keeps tokens out of route signatures
      - Internally uses ProviderHandler to derive provider services and latest tokens
    - SpecifyUserContext (admin/sandbox/internal flows): openhands/app_server/user/specifiy_user_context.py
      - Request-scoped override carrying a specific user_id (or None for admin)
  - Injectors:
    - AuthUserContextInjector provides request-scoped UserContext from request-auth.

Observations
- agent-sdk/agent-server have no concept of user_id; user is an app concern.
- app-server introduces persistence for conversation metadata in DB (SQLAlchemy models) with user_id columns, but these are used via services that consume UserContext.
- New V1 routes do not carry user_id in path or query; scoping is enforced inside services via UserContext (see SQLAppConversationInfoService._secure_select()).

Auth/token threading in V1 app-server
- Tokens are not threaded through route signatures.
- Provider tokens are accessed via UserContext methods, often converted to SecretSource for the agent runtime.
- For agent-server communication, a short-lived session API key is used via header X-Session-API-Key.

Org risk and recommendation
- Enterprise is adding organizations (org_id) linking users to orgs. Do NOT introduce org_id into routes.
- Extend context instead of paths/signatures:
  - Option A: add optional org_id to UserContext.get_user_info() result and to SQL filter logic; record created_by_org_id in metadata.
  - Option B: introduce a separate OrganizationContext resolved by DI and consumed by services alongside UserContext.
- Ensure all service-layer DB queries filter by (user_id OR memberships via org_id) based on policy. Keep route signatures unchanged.

Actionable follow-ups
- Add org-aware filtering in SQL services without affecting routes.
- Keep provider tokens and org scoping behind UserContext/OrganizationContext.

Key references
- user_context.py, auth_user_context.py, specifiy_user_context.py
- app_conversation/sql_app_conversation_info_service.py (uses get_user_id() for row-level filtering)
- event/filesystem_event_service.py (permissioning via app_conversation_info_service)


Audit summary
- Routers
  - /api/v1/users: user_router.py uses DI UserContext.get_user_info(); no user_id exposure
  - /api/v1/app-conversations: app_conversation_router.py uses service injectors; stream-start pins UserContext in InjectorState for streaming; no user_id exposure
  - /api/v1/events: event_router.py uses EventService via DI; no user_id exposure
  - /api/v1/webhooks: webhook_router.py validates sandbox via X-Session-API-Key, uses as_admin() for webhook auth, and resolves per-user context via injector.get_for_user(user_id) for JWS-secret fetch; no user_id in route signatures
- Services
  - SQLAppConversationInfoService scopes by UserContext.get_user_id() within _secure_select(), count, get, batch_get, save
  - SQLAppConversationStartTaskService injector resolves user_id once from UserContext and binds it to service instance; no route exposure
  - LiveStatusAppConversationService uses UserContext for authorship/tokens; interacts with Agent Server using sandbox session API key
- Legacy shims still mounted and used in V1 flows
  - server/routes/conversation.py and manage_conversations.py continue to use Depends(get_user_id); treat them as compatibility shims only; avoid new feature work there

---

## 2) ConversationPaths

Status
- agent-sdk and agent-server store conversation data; app-server keeps metadata in DB and events in filesystem.
- No user_id is embedded in filesystem paths in V1 app-server.
  - FilesystemEventService stores events under: {persistence_dir}/v1/events/{conversation_id}/timestamp_kind_eventId
- Therefore, V0 concern about user_id in conversation paths is not present in V1 app-server.

Validation
- FilesystemEventService._ensure_events_dir(), _get_event_files_by_pattern() demonstrate directory layout without user_id.
- All user scoping happens at service layer by checking conversation ownership via DB service (batch_get_app_conversation_info) before returning file-based events.

Recommendation
- Keep user/org scoping out of paths. If orgs are added, continue enforcing access via DB/service checks, not via path namespacing.

Key references
- app_server/event/filesystem_event_service.py
- app_server/app_conversation/sql_app_conversation_info_service.py

---

## 3) TokenSource → LookupSecret(SecretSource)

Status
- agent-sdk’s equivalent boundary is SecretSource with implementations:
  - StaticSecret for literal values
  - LookupSecret for remote fetch

Token sweep (direct usage in routes/services)
- V1 app-server
  - No provider tokens in route signatures. Provider access goes through UserContext → ProviderHandler and is consumed as SecretSource (StaticSecret or LookupSecret).
  - X-Session-API-Key appears only for agent-server calls (headers) and sandbox auth in webhooks; not a provider token. Required to authorize runtime access.
  - X-Access-Token (JWS) appears only in webhook secret fetch flow; scoped to user_id and provider_type (and can include org_id in future). Not required in public routes; only used by sandboxes to retrieve secrets.
- Legacy routes still mounted
  - /api/user/* expects provider_tokens via Depends(get_provider_tokens) plus access_token (legacy external auth) and user_id. These are the only places exposing provider token inputs at the REST surface today.
  - Guidance: keep them as compatibility endpoints. Do not add new surfaces that accept provider tokens. Prefer V1 SecretSource/JWS approach.
- Services
  - SQL services do not pass provider tokens; they consume UserContext.user_id for row-level filtering only.
  - LiveStatusAppConversationService uses JWT service to create X-Access-Token for LookupSecret and uses sandbox X-Session-API-Key to talk to agent-server; no provider tokens in signatures.

Access token in routes?
- We do not need an access token in public V1 routes. The JWS access token (X-Access-Token) is strictly an internal sandbox-to-app-server credential for GET /api/v1/webhooks/secrets.
- For user-initiated calls, standard app auth + DI is sufficient; provider tokens should never be threaded through public route signatures.

- app-server V1 maps provider tokens into SecretSource:
  - In LiveStatusAppConversationService._build_start_conversation_request_for_user():
    - If web_url is configured, constructs a LookupSecret to GET /api/v1/webhooks/secrets with X-Access-Token (JWS) that includes user_id and provider_type (scoped and expirable)
    - Else falls back to StaticSecret with latest provider token from UserContext
- Therefore, token refresh/leakage is solved at the boundary: agent runtime calls back via LookupSecret; route signatures remain free of tokens.

Other tokens/headers in signatures
- X-Session-API-Key: agent-server session API key for starting conversations via app-server → agent-server POSTs.
- X-Access-Token: app-server-issued JWS for sandbox to retrieve secrets via /api/v1/webhooks/secrets.
- No provider_tokens or user_id appear in public REST route signatures.

Recommendation
- Keep all external-provider token logic behind SecretSource and JWT/JWS.
- With orgs, add org_id into the JWS claims when appropriate, and enforce in webhook secret retrieval by validating both user and org scopes.

Key references
- app_conversation/live_status_app_conversation_service.py (GIT_TOKEN secret construction)
- event_callback/webhook_router.py (GET /secrets; validates JWS and fetches provider tokens via per-user DI)
- user/auth_user_context.py (ProviderHandler usage is internal)

---

## V1 REST Routes (current)

Mounted under /api/v1 via openhands/app_server/v1_router.py

- /api/v1/app-conversations (app_conversation_router)
  - GET /search: filter by title/created_at/updated_at; pagination via page_id, limit
  - GET /count: same filters, returns count
  - GET /: batch get by ids[]=UUID
  - POST /: start task (returns AppConversationStartTask); uses background processing and X-Session-API-Key to talk to agent-server
  - POST /stream-start: streams AppConversationStartTask updates until READY/ERROR
  - GET /start-tasks/search: filter by conversation_id; sort order; pagination
  - GET /start-tasks/count
  - GET /start-tasks: batch get by ids[]=UUID

- /api/v1/events (event_router)
  - GET /search: filters (conversation_id, kind, timestamp ranges), sort, pagination
  - GET /count: same filters, count
  - GET /: batch get by id[]=str (UUIDs)

- /api/v1/sandboxes (sandbox_router)
  - GET /search: pagination
  - GET /: batch get by id[]=str
  - POST /: start sandbox (optional sandbox_spec_id)
  - POST /{sandbox_id}/pause
  - POST /{sandbox_id}/resume
  - DELETE /{id}: delete sandbox (NB: path parameter name differs from function param; consider standardizing to {sandbox_id})

- /api/v1/webhooks (webhook_router)
  - POST /{sandbox_id}/conversations: upsert conversation info from agent-server callback
  - POST /{sandbox_id}/events/{conversation_id}: append events from agent-server callback
  - GET /secrets: return provider secret value for scoped JWS (X-Access-Token)

- /api/v1/users (user_router)
  - GET /me: returns current authenticated user info

Auth primitives in routes
- No user_id in REST paths.
- No provider_tokens in route signatures.
- Authentication/authorization via DI-provided contexts and header tokens (session or access) where necessary.

---

## Legacy/V0 surface still present (for UI compatibility)

Below is a concise mapping of legacy endpoints that the frontend still calls, and how they relate to V1. Some are intentionally retained because they provide functionality not yet covered by the new V1 surfaces. All items verified against the current codebase.

- /api/user/* (legacy Git/user provider APIs)
  - Implemented in: openhands/server/routes/git.py
  - Frontend usage: frontend/src/api/git-service/*.ts and suggestions-service.api.ts
  - Purpose: repository discovery, branches, installations, microagents listing/content, suggested tasks
  - Status for V1: Retained. These serve as integration-oriented endpoints for git providers and repo scanning. No direct V1 replacement yet. They remain part of the V1 experience and are mounted alongside /api/v1.

- /api/conversations/{conversation_id}/events and related conversation endpoints (legacy conversation runtime APIs)
  - Implemented in: openhands/server/routes/conversation.py
  - Frontend V1 usage: frontend/src/api/conversation-service/v1-conversation-service.api.ts uses POST /api/conversations/{conversationId}/events to send messages

Org policy without schema changes
- Scope: Avoid org_id in routes or schemas. Apply org-aware access exclusively in DI/service layer and JWS claims.
- Strategy: Compute allowed_user_ids for the active context. Services filter by created_by_user_id in StoredConversationMetadata using either:
  - user_id from UserContext when no org is active, or
  - a set of user_ids derived from OrganizationContext memberships when an org is active.
- No table changes are required because we filter on user_id. Organization membership is resolved externally (e.g., enterprise DB) and injected via DI.

DI structure
- OrganizationContext (DI, optional)
  - active_org_id: str | None  (selected org or None)
  - member_user_ids(): set[str]  (users in selected org)
  - is_member(user_id: str) -> bool
- Default OH implementation returns no active_org_id and empty membership, so behavior equals current user-only scoping.
- Enterprise can provide an OrganizationContext injector that reads org selection from auth/session and resolves memberships from its own DB.

Service scoping wrappers (no schema change)
- Wrap SQLAppConversationInfoService with OrgScopedAppConversationInfoService:
  - search/count/get/batch_get: if active_org_id, filter where user_id IN member_user_ids(); else filter by current user_id.
  - save: assert created_by_user_id belongs to current user or org membership (policy: allow only current user as author; org membership governs visibility, not authorship).
- Wrap SQLAppConversationStartTaskService similarly using created_by_user_id.
- Event access: EventService should validate visibility via AppConversationInfoService prior to reading from filesystem; reuse the same org-aware filter.

Pseudocode (service wrapper)
- def _user_ids_scope():
  - org_ctx = OrganizationContext(); user_ctx = UserContext()
  - if org_ctx.active_org_id: return org_ctx.member_user_ids()
  - else: return { await user_ctx.get_user_id() }
- For queries: query.where(StoredConversationMetadata.user_id.in_(scope))

JWS claim integration (secrets)
- X-Access-Token JWS may optionally include org_id claim when an org is active.
- GET /api/v1/webhooks/secrets should verify org_id claim via OrganizationContext and enforce that the requested secret belongs to a user within org membership.
- This keeps org scope entirely in DI/JWS logic; no route or schema change needed.

Legacy endpoints
- /api/user/* remain as compatibility surfaces and accept provider_tokens/access_token today; do not add org_id to their routes. If org-aware behavior is needed there, enterprise can wrap or replace them with V1 endpoints that resolve provider tokens via DI and JWS instead of raw provider_tokens.

Risks
- Reintroducing user_id or org_id into route signatures through new endpoints. Mitigate by requiring DI-only scoping for all new features.
- Inconsistent scoping across services. Mitigate by centralizing scope computation (_user_ids_scope) and reusing it in all service wrappers.
- Performance on IN clauses with large orgs. Mitigate with caching of member_user_ids and pagination limits; optionally implement server-side membership expansion via join in enterprise layer.

Rollout plan
- Phase 1 (no schema changes):
  - Implement OrganizationContext injector (default no-op in OH; enterprise provides real one).
  - Add OrgScoped wrappers for AppConversationInfoService, AppConversationStartTaskService, and EventService.
  - Add optional org_id claim to JWS token issuance path in LiveStatusAppConversationService and verify in webhook_router.get_secret.
- Phase 2: Monitor perf; if needed, enterprise may introduce derived indices or membership-materialized views on its side without touching OH schemas.

  - How V1 wiring works:
    - get_remote_runtime_config at GET /api/conversations/{conversation_id}/config detects V1 sessions (UUID) and maps to sandbox_id + session_api_key so that a V1 session can still use these endpoints
    - add_event and add_message forward events to the appropriate runtime (legacy or mapped V1)
  - Status for V1: Retained. These are part of the V1 runtime interaction shim and are intentionally kept to avoid duplicating runtime event endpoints under /api/v1.

- /api (legacy conversation management)
  - Implemented in: openhands/server/routes/manage_conversations.py
  - Purpose: V0 session lifecycle management and metadata; has shims to include V1 results when possible
    - Example: GET /api/conversations/{conversation_id} tries V1 via AppConversationService first, else falls back to V0
    - POST /api/conversations creates V0 sessions (distinct from /api/v1/app-conversations which starts a V1 conversation via agent-server)
  - Status for V1: Mixed. Some endpoints act as shims for compatibility and aggregation; new V1 creation flows are at /api/v1/app-conversations. These legacy endpoints are mounted and available but should be considered compatibility layers.

Notes
- Routing: openhands/server/app.py conditionally mounts v1_router alongside legacy routers when server_config.enable_v1 is true. The frontend calls a mix of /api/v1/* and legacy /api/* or /api/user/* where necessary. This is expected during the transition.
- Frontend usage verification:
  - frontend/src/api/conversation-service/v1-conversation-service.api.ts
    - sendMessage() posts to /api/conversations/{id}/events (legacy path) using V1 conversationUrl/session key
    - getVSCodeUrl(), pauseConversation(), and uploadFile() talk directly to the agent-server via conversationUrl (endpoints provided by agent-server)
  - frontend/src/api/suggestions-service/suggestions-service.api.ts calls /api/user/suggested-tasks
- Recommendation: keep these legacy endpoints stable while we evaluate which ones should get V1-native equivalents. Where legacy endpoints are used in V1 flows (e.g., sending messages), they are effectively part of the V1 surface and should be documented as such.
- Org risk: even for retained legacy endpoints, avoid adding org_id to paths. Apply org scoping via DI/service checks and JWS where needed.

---

## Open risks and proposed refactor plan (orgs)

1) Data model
- Do not add org_id to OpenHands models or schemas at this stage.
- Keep storage keyed by user_id only; org-aware access should be enforced purely via DI/service policy (UserContext/Organization policy), never via new columns or route params.

2) Context and DI
- Option A: Extend UserContext.get_user_info() to include org memberships and an active org_id (if selected).
- Option B: Add OrganizationContext via DI and inject alongside UserContext. Services consult both for scoping.

3) Service filters
- Update SQLAppConversationInfoService._secure_select() to filter by either user_id or org memberships as policy dictates. Avoid touching route signatures.

4) Secret scoping
- When issuing JWS for /webhooks/secrets, include org_id claim and verify it on fetch. Ensure provider token resolution respects org policies.

5) Endpoint contracts
- Keep V1 routes as-is (no org_id in path). Avoid proliferating new route variants.

6) Cleanup
- Gradually remove V0 routes once frontend migrates fully to V1 equivalents.

---

## Quick pointers (file paths)
- V1 router aggregator: openhands/app_server/v1_router.py
- UserContext: openhands/app_server/user/user_context.py
- AuthUserContext: openhands/app_server/user/auth_user_context.py
- Admin override: openhands/app_server/user/specifiy_user_context.py
- V1 DB services: app_conversation/sql_app_conversation_info_service.py, sql_app_conversation_start_task_service.py
- Events: app_server/event/filesystem_event_service.py
- Secrets/lookup: app_server/event_callback/webhook_router.py, app_conversation/live_status_app_conversation_service.py

This file is a living document – update as implementations evolve.
