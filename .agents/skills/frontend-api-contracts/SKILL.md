---
name: frontend-api-contracts
description: This skill should be used when the user asks to "add an API call", "change a backend", "update settings persistence", "change conversation events", "fix backend auth", "add Agent Server support", or changes src/api, backend registry, Cloud/runtime transport, settings, secrets, compatibility, or conversation resume behavior.
---

# Frontend API and Backend Contracts

Keep Canvas as a typed consumer of Agent Server and Cloud contracts. Preserve repository ownership, compatibility floors, transport boundaries, authentication modes, and durable state invariants.

## Workflow

1. Read `references/guide.md` before modifying frontend API access, backend selection, settings, secrets, Agent Server compatibility, or Cloud runtime behavior.
2. Determine whether the contract belongs in `software-agent-sdk` and its TypeScript client before adding Canvas code.
3. Use typed clients and centralized option builders for Agent Server/runtime calls; use `callCloudProxy` only for Cloud App-API calls.
4. Raise `compatibility.minimumAgentServer` when Canvas begins requiring new server behavior.
5. Verify Local and Cloud paths, authentication, persistence transitions, and affected conversation modes.
6. Run the direct-call guard and focused API/state tests.

## Reference

- **`references/guide.md`** — API transport rules, allowed exceptions, published-client constraints, compatibility conventions, backend registry and auth behavior, settings/secrets persistence, backend UI invariants, and Cloud resume gating.
