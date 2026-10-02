# Runtime provider selection proposal

Status: proposed for discussion. Relates to [#17833](https://github.com/OpenHands/OpenHands/issues/17833). This document does not add a runtime provider or define a released wire contract.

## Decision proposed

Let an Agent Server advertise the runtime options it can provision. Allow a new conversation to request one of those options while leaving the backend default unchanged. Persist the resolved choice on the server so reopening a conversation never depends on a browser preference. Introduce local and Docker choices first. Add hosted providers only after the same lifecycle and workspace rules work for those two choices.

The first contribution is this Canvas integration proposal. Canonical models, provisioning and TypeScript-client changes belong in `OpenHands/software-agent-sdk`. The endpoint and field names below are suggestions for SDK review, not types to copy into Canvas.

## Current behavior and the gap

This baseline was checked against Canvas `2414d6ee5` and Agent Server 1.50.1 on 2026-10-02.

| Existing behavior                                                                                                                                                                     | Source                                                                                                                                                                               | Consequence                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OH_CONVERSATION_RUNTIME=docker` starts new conversations in separate containers. Existing local conversations are not converted. Conversations can still share host workspace files. | [README, Option 3](https://github.com/OpenHands/OpenHands/blob/2414d6ee5/README.md#option-3-with-multiple-docker-sandboxes)                                                          | Container isolation is not a promise of separate files.                                                                                                             |
| Workspace resolution inspects `ServerInfo.conversation_runtime` and rejects unsupported host workspace selections in isolated mode.                                                   | [usesIsolatedWorkspace and resolveNewConversationWorkspace](https://github.com/OpenHands/OpenHands/blob/2414d6ee5/src/api/conversation-workspace.ts#L18)                             | A picker must resolve workspace rules for the chosen runtime, not just the backend mode.                                                                            |
| Conversation builders also contain an `execution_runtime` path that chooses `LocalWorkspace` or `DockerExecutionWorkspace`.                                                           | [buildConfiguredConversationSettings](https://github.com/OpenHands/OpenHands/blob/2414d6ee5/src/api/agent-server-adapter.ts#L1126)                                                   | `execution_runtime` and `conversation_runtime` must not be assumed to describe the same thing. The SDK must settle the authority before Canvas removes either path. |
| The runtime response exposes `runtime_status`, `can_resume` and `runtime_error`; the client already has runtime lookup and reprovision operations.                                    | [ConversationRuntimeInfo](https://github.com/OpenHands/software-agent-sdk/blob/cc97bf234ea63b9276a7bd71ef81491f3849a7b2/openhands-agent-server/openhands/agent_server/models.py#L48) | Extend existing lifecycle semantics instead of creating a second status machine in Canvas.                                                                          |
| File, terminal and git requests use conversation-scoped runtime routing.                                                                                                              | [RuntimeRouter](https://github.com/OpenHands/software-agent-sdk/blob/cc97bf234ea63b9276a7bd71ef81491f3849a7b2/openhands-agent-server/openhands/agent_server/runtime_router.py#L65)   | Provider selection must preserve these routes and their authorization boundary.                                                                                     |

Neither a backend URL selector nor a different workspace discriminator is sufficient to choose a hosted provider per conversation. Backend selection chooses the control plane and its identity. Runtime selection chooses where that control plane executes one conversation.

## RPS-001: SDK contract required before a Canvas picker

The SDK should own these operations and publish browser-compatible TypeScript methods. Canvas must not call suggested paths until a released client supports them.

### Discovery

Proposed capability: `runtime_selection_v1`. Proposed discovery operation: `GET /api/runtime-options` through a typed client method. The authenticated response describes options available to the caller on this backend:

```json
{
  "default_option_id": "local",
  "options": [
    {
      "id": "local",
      "label": "Local",
      "provider_id": "builtin.local",
      "availability": "available",
      "unavailable_reason": null,
      "supported_agent_kinds": ["openhands", "acp"],
      "workspace_modes": ["host_directory"],
      "supports_parent_workspace": true
    },
    {
      "id": "docker",
      "label": "Docker",
      "provider_id": "builtin.docker",
      "availability": "unavailable",
      "unavailable_reason": "Docker is not running",
      "supported_agent_kinds": ["openhands"],
      "workspace_modes": ["isolated"],
      "supports_parent_workspace": false
    }
  ]
}
```

These values illustrate a response, not a claim about current ACP or Docker support. IDs are stable opaque identifiers scoped to the backend. Options represent configured provisioning choices; multiple options may use the same provider with different server-managed configuration. Credentials, arbitrary endpoints and raw provider configuration must not appear in discovery. The server filters options by caller permissions and revalidates them during creation.

### Creation and resolved identity

Propose an optional `runtime_option_id` in the canonical conversation-create model. Omission preserves the server's default behavior for old clients. A supplied ID is an explicit request: an unknown, forbidden, unavailable or incompatible choice must fail with a typed error and must never silently fall back to local execution.

The SDK validates the selection together with the workspace request, agent kind and parent conversation. Existing workspace payloads remain the source of requested workspace configuration; no competing Canvas-only workspace schema is added. Unsupported host paths, worktrees or repository choices fail before provisioning. For isolated workspaces the server resolves the internal path and returns it in the existing workspace response.

Persist the resolved option and provider identity with the conversation. Expose that identity on conversation/runtime reads through an SDK-owned type. Selection is immutable after creation in the first release. Reprovisioning restores the same option and workspace; it is not provider migration. Changing a default or deleting a configured option cannot silently relabel an existing conversation.

The server owns allocation and rollback. A failed creation must either clean up its allocation or retain a recoverable record with an error; retries must not leak containers or create duplicate paid resources. Reuse existing conversation IDs and creation retry semantics where possible. The SDK must specify how duplicate create requests return the same result before hosted provisioning is enabled.

### Lifecycle and credentials

Reuse `ConversationRuntimeInfo` and existing runtime/reprovision operations. The server remains authoritative for `available`, `starting`, `missing`, `ownership_lost` and `error`, including whether recovery is allowed. Authentication failures and permission failures are not ordinary provisioning retries. Errors exposed to Canvas must omit credentials and sensitive provider diagnostics.

Provider credentials belong to the server's credential/configuration mechanism. Canvas supplies only a configured option ID. Browser access continues through existing authenticated conversation routes or an SDK-supported runtime URL with scoped session credentials. Selecting a provider must not enable unauthorized cross-conversation routing or introduce an arbitrary browser-supplied proxy destination. Runtime options must disclose their workspace boundary: local execution is not a shell sandbox and explicitly shared workspaces remain shared.

Deleting an option prevents new selection. Existing conversations retain their identity and show an explicit unavailable/recovery state if the provider can no longer serve them. Removing configuration must not delete conversation data or provider volumes implicitly. Cleanup, retention and credential rotation remain server-owned operations.

## RPS-002: Canvas ownership and interaction

Add runtime selection to new-conversation setup only after discovery is supported. Show the backend's default as the initial choice with the runtime label and a short explanation of workspace isolation. Disable unavailable or incompatible options with the server-provided reason. Validate repository/path choices when the selection changes and explain any incompatible value before the user starts the conversation.

Keep the pending selection in the existing new-conversation form owner. Do not add a second persisted runtime default in browser storage. Clear the pending choice when the backend changes and reload that backend's options. Preserve the user's selection across recoverable form errors. Refresh discovery after a selection-related rejection and require a new deliberate choice if the option disappeared.

Send the selected ID once through the shared conversation builder. Render the resolved server choice after creation. Loading an existing conversation, changing the active backend or reprovisioning must never apply the current form default to that conversation.

| Consumer                                     | First-release rule                                                                                                                                                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local backend, standalone or embedded Canvas | Share discovery and create integration. No browser-global default. Embedders retain the existing backend configuration boundary.                                                                               |
| Cloud backend                                | Keep the existing Cloud provisioning flow until its App API supports the same selection semantics. Do not send new SDK fields to an unsupported Cloud create endpoint.                                         |
| OpenHands and ACP                            | Filter by SDK-advertised agent support. Never infer support from a provider name.                                                                                                                              |
| Planner and delegated conversations          | Inherit the parent's resolved runtime/workspace policy server-side. If the runtime cannot provide the required shared workspace, reject that child flow with a reason. Do not start it on the backend default. |
| Existing conversations                       | Keep persisted runtime and workspace identity. Unknown historical identity is displayed as unknown until the server can resolve it.                                                                            |
| Automations                                  | Keep current behavior until `OpenHands/automation` can persist, validate and dispatch the runtime selection. Do not make scheduled execution depend on browser state.                                          |

## RPS-003: Staged rollout and compatibility

1. **SDK local/Docker contract.** Agree on the canonical mode terminology, discovery model, create field and resolved identity. Implement authorization, persistence, workspace validation and recovery tests in `software-agent-sdk`. Preserve old-client omission behavior.
2. **Published TypeScript client.** Export the SDK-derived types and typed discovery/create/read methods. Cover payload serialization, structured errors and old-server capability absence. No Canvas-local copies of wire types.
3. **Canvas conversation picker.** Consume the released client for supported local backends. Hide the picker on old servers and preserve their existing create flow. Capability checks must precede sending new fields. Increase `compatibility.minimumAgentServer` when the feature begins requiring the new behavior, using the first compatible released version rather than a guessed version.
4. **Cloud and automation integration.** Their owners adopt the same semantics through their existing APIs. Add persistence and dispatch coverage before enabling those selectors. A changed default must not change already scheduled runs' recorded selection.
5. **Hosted providers.** Add server-side adapters behind the same contract. Require isolation, reconnect, reprovision, cancellation, credential handling and cost/retention tests before exposing a provider in Canvas.
6. **Discovery/catalog experience.** Consider a curated provider catalog only after configuration and lifecycle are proven. An entry must disclose who operates the environment, what data leaves the host, billing ownership and retention. Installing a catalog entry must not execute unreviewed provider code in the browser.

Rollback hides new selection on unsupported backends without rewriting stored conversation identity. A failed capability probe is not permission to reinterpret an explicit selection. If no choice has been made, the existing default create path may remain available; if a choice has been made and cannot be validated, block submission with retry.

## Acceptance and verification for follow-up implementation

These items are proposed acceptance criteria, not claims that this PR implements them.

- [ ] One backend can start a local conversation and a Docker conversation concurrently without changing an instance-wide setting.
- [ ] Each conversation retains its resolved option across browser reload, backend switch, server restart and permitted reprovisioning.
- [ ] Unsupported or unavailable selection fails without fallback or orphaned allocations. Repeating a creation request does not provision twice.
- [ ] Existing clients that omit the field preserve current behavior. Existing servers do not receive unsupported fields.
- [ ] Workspace validation covers host directories, isolated filesystems and the planner/delegation sharing contract. Separate containers never imply separate files without evidence.
- [ ] Runtime requests remain authorized per conversation. Tests reject unauthorized cross-conversation routing and access outside the workspace boundary enforced by the selected runtime. Explicitly shared workspaces remain accessible to authorized participants. Discovery/errors contain no provider secrets.
- [ ] Canvas tests cover selection, unavailable options, backend changes, stale discovery, creation failure and server-resolved identity. Exercise both standalone and embedded entry points where their initialization differs.
- [ ] Cloud and automation selectors remain absent until their respective API paths pass equivalent tests.

## Decisions requested from maintainers

Confirm local/Docker per-conversation selection as the first implementation slice, the SDK as contract owner and the proposed inheritance rule for planner/delegated workspaces. Decide whether the SDK can extend its current runtime lifecycle response or needs a separate configuration resource. Hosted-provider adapters, provider migration, marketplace installation and automation scheduling changes are explicitly outside the first slice.

This proposal makes no claim that a marketplace or a specific hosting vendor is approved. It gives the SDK and Canvas owners a contract boundary to review before UI implementation.
