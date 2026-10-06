# Codex onboarding evidence and credential reuse proposal

Status: review proposal; cross-flow credential reuse is not implemented by the current draft.

## Decision

Keep the existing ChatGPT subscription OAuth flow as the login owner. Extend the SDK/Agent Server to retain the complete authorization response and serve both the SDK LLM transport and Codex ACP from one encrypted, versioned server record. Canvas should reuse one login presentation and safe status contract.

The new value is Codex ACP onboarding and credential provisioning on a server without a host CLI login. ChatGPT OAuth and API-key-free Codex usage already exist. This proposal must not be described as a new OAuth capability.

## Evidence now available

1. [Transport-fixture onboarding video](onboarding-auth-fixture.webm): a fresh browser context selects Codex, displays a nonfunctional device code, copies/cancels it, connects on a second attempt, enables Next, reaches Say hello, and closes without requesting a model.
2. [Real existing-account onboarding video](onboarding-existing-login.webm): the original, packaged local Agent Server recognizes the previously human-authorized Codex account, permits Next with the API-key field empty, reaches Say hello, and persists the Codex ACP selection.
3. [Fixture observations](onboarding-auth-fixture-results.json) and [real-server observations](onboarding-existing-login-results.json) contain safe HTTP method/path/status observations, not request bodies, headers or credentials.

Both recordings show the real built Canvas and packaged Agent Server. The fixture recording replaces only OpenAI authentication transport. The existing-account recording uses the original encrypted Codex credential; it does not initiate a fresh login. Neither recording starts a model turn or proves that an LLM Profile login is reused by ACP.

Sources: [onboarding card](https://github.com/luxleader/OpenHands/blob/cefa2a1d49c28c9849fb9a93cd96d811c9cd4cb9/src/components/features/onboarding/steps/setup-acp-secrets-step.tsx#L172), [existing SDK transport](https://github.com/OpenHands/software-agent-sdk/blob/aae9c437f029d45d7a64924285015d3da87ade94/openhands-sdk/openhands/sdk/llm/auth/openai.py#L713), [draft ACP transport reuse](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-agent-server/openhands/agent_server/codex_auth.py#L112).

## Current boundary

| Existing LLM Profile login                                           | Current ACP draft                                                                      |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| /api/llm/subscription/openai/\* delegates to OpenAISubscriptionAuth. | /api/acp/codex/auth/\* delegates to CodexAuthService.                                  |
| CredentialStore retains access_token, refresh_token and expires_at.  | CodexAuthFile retains ID/access/refresh tokens for auth.json.                          |
| SDK LLM transport consumes OAuthCredentials.                         | ACP receives CODEX_AUTH_JSON through the existing versioned file-credential lifecycle. |
| LLM login does not currently populate the ACP credential.            | ACP login does not currently populate the LLM credential.                              |

Sources: [LLM router](https://github.com/OpenHands/software-agent-sdk/blob/aae9c437f029d45d7a64924285015d3da87ade94/openhands-agent-server/openhands/agent_server/llm_router.py#L96), [OAuth credential model](https://github.com/OpenHands/software-agent-sdk/blob/aae9c437f029d45d7a64924285015d3da87ade94/openhands-sdk/openhands/sdk/llm/auth/credentials.py#L32), [Codex credential model](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-agent-server/openhands/agent_server/codex_auth.py#L44).

## Proposed ownership and adapters

- Scope the shared record to one authenticated Agent Server user/store. Never reuse credentials across selected backends or the separate Cloud App API.
- Retain the provider's real ID token, access token, refresh token, expiry and account metadata in one encrypted FileSecretsStore record. Tokens remain server-side and use the repository's Pydantic secret helpers.
- Inject a server-backed credential adapter into OpenAISubscriptionAuth instead of letting Agent Server LLM routes own a separate file store.
- Supply Codex auth.json as an adapter projection of that same record through VersionedCredentialBinding. Existing CODEX_AUTH_JSON references remain a compatibility projection, not a second independently writable token record.
- The login UI and pending device attempt have one server owner. Both consumer endpoints delegate to it and continue to return only safe status, a user code and an opaque polling handle.
- Preserve standalone SDK CredentialStore behavior. The server integration must not silently change public OAuthCredentials field types or remove existing SDK APIs.

Sources for the reusable mechanisms: [OpenAISubscriptionAuth injection](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-sdk/openhands/sdk/llm/auth/openai.py#L461), [FileSecretsStore](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-agent-server/openhands/agent_server/persistence/store.py#L425), [versioned reads/writes](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-agent-server/openhands/agent_server/persistence/store.py#L652), [binding contract](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-sdk/openhands/sdk/credential.py#L44).

## Old credentials and migration

Existing LLM credentials lack a persisted ID token. Do not fabricate one or claim that copying access/refresh fields creates a valid Codex login.

1. Load and validate the old private record on the server. Migrate it once into the canonical encrypted record using compare-and-set; preserve all usable LLM fields.
2. Retain an ID token if a normal provider refresh actually returns one. Otherwise keep the LLM connected and ask for ChatGPT reconnection only when the user enables Codex.
3. A complete existing Codex credential can be imported only for the same user/backend and projected into SDK LLM form. Avoid two records independently refreshing the same authorization.
4. Preserve explicit disconnected state across restart. Automatic host-file import must not resurrect a connection that the user disabled.
5. Never delete or rewrite an unrelated host CLI login as part of migration.

The missing-ID-token condition is grounded in the current OAuthCredentials model and \_complete_device_login, which retains only access/refresh tokens and expiry. Migration and shared storage are proposed SDK work, not behavior proven by these recordings.

## Reuse and disconnect behavior

The shared server record needs explicit consumer enablement for SDK LLM and Codex ACP. These are server-owned connection states, not additional browser token copies.

| User action                                        | Required behavior                                                                                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Enable Codex after a compatible LLM login          | Offer reuse of the connected account; grant ACP access without a second OpenAI authorization.                                    |
| Enable Codex with an old incomplete LLM credential | Preserve LLM access; request reconnection to obtain the missing provider-issued ID token.                                        |
| Disconnect Codex                                   | Disable ACP provisioning and invalidate its credential binding; keep LLM access if it is still enabled.                          |
| Disconnect LLM subscription                        | Disable that consumer; do not unexpectedly disconnect a still-enabled Codex agent.                                               |
| Sign out of ChatGPT everywhere on this server      | Explicitly disable both consumers, cancel pending/refresh work and remove the canonical credential.                              |
| Disable the last consumer                          | Remove the token record while retaining a disconnected tombstone/revision so stale work or host auto-import cannot reconnect it. |

These consumer states are a proposal. Current per-flow logout APIs are not already coordinated in this way.

## Refresh and concurrency gate

A shared rotating refresh token cannot safely have two independent owners. Compare-and-set rejects stale writes, but alone does not prevent concurrent provider refreshes from rotating or invalidating the same token.

The server must coordinate SDK refresh and the native Codex process's auth-file refresh/writeback, using the versioned binding and a shared refresh ownership/lease mechanism. Reprovision or restart affected ACP sessions when a credential revision changes, rather than leaving an old in-memory credential usable indefinitely.

Do not enable cross-flow token sharing until actual pinned codex-acp behavior verifies that refresh ownership, native file writeback, cancellation and logout cannot race. If that cannot be guaranteed, reuse the UI/transport but keep explicitly separate authorizations and explain the tradeoff; do not silently copy a rotating token into two independent stores.

Existing integration points: [ACP file lifecycle](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-sdk/openhands/sdk/agent/acp_file_credentials.py#L125), [credential restart hook](https://github.com/luxleader/software-agent-sdk/blob/f3a5a8f00d81a0b57cbd8e2fd6907e1dbee2c77d/openhands-sdk/openhands/sdk/agent/acp_agent.py#L2031).

## Implementation and release order

1. SDK/Agent Server: full-response retention, encrypted owner/adapter, migration, consumer disconnect rules, and refresh concurrency. Keep old LLM endpoints compatible; make any ACP routes thin consumer adapters.
2. TypeScript client: typed safe status/challenge contracts for the shared owner, with no token-bearing browser types.
3. Canvas: reuse one login component and status owner across LLM Profile, Codex onboarding and Agent settings; show the already-connected account and an explicit reuse action where necessary.
4. Publish official compatible SDK/client/server releases, update exact Canvas pins and minimumAgentServer, then complete real-account/Docker/remote acceptance before moving the PR out of draft.

No release version is invented. Maintainer agreement on shared credential ownership is required before this architectural follow-up.

## Acceptance matrix for the follow-up

- LLM login to Codex reuse and Codex login to LLM reuse, with no second authorization when the credential is complete.
- Old LLM credential without ID token stays usable; Codex reconnection is explicit.
- Backend/account separation and browser reload/server restart preserve the intended consumer state.
- Simultaneous SDK refresh/native ACP rotation rejects stale writes and does not invalidate the latest authorization.
- Per-consumer disconnect, global sign-out, stale login completion and host-file import cannot resurrect revoked access.
- Existing standalone SDK/API-key/manual-file alternatives remain usable.
- HTTP responses, browser storage, logs and published artifacts contain no OAuth credentials.
- Real first-run consent/model acceptance remains separate from the fixture demonstration.
