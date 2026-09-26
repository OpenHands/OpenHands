---
name: api-contracts
description: >
  The two strict REST-access conventions for the frontend — agent-server
  calls go through @openhands/typescript-client, Cloud routes go through
  callCloudProxy — enforced by CI. Load before making REST calls or editing
  src/api/.
triggers:
  - api access
  - agent-server call
  - typescript-client
  - cloud proxy
  - session api key
  - rest call
---

## API Access Rules

Two strict conventions govern every REST call in the frontend. Violations break CI
via `src/api/no-direct-agent-server-calls.test.ts`.

### Rule 1 -- Agent-server calls must use `@openhands/typescript-client`

All calls that target the local agent-server (`/api/*`, `/server_info`, `/sockets`)
**must** go through typed client classes from `@openhands/typescript-client`, **never**
raw `axios`, `fetch`, or the legacy shared `openHands` axios instance.

Available clients and their subpath imports:

- `ConversationClient` -- `@openhands/typescript-client/clients`
- `FileClient` -- `@openhands/typescript-client/clients`
- `VSCodeClient` -- `@openhands/typescript-client/clients`
- `ServerClient` -- `@openhands/typescript-client/clients`
- `RemoteWorkspace` -- `@openhands/typescript-client/workspace/remote-workspace`
- `RemoteEventsList` -- `@openhands/typescript-client/events/remote-events-list`

Client options are always assembled via helpers in `src/api/agent-server-client-options.ts`:

- `getAgentServerClientOptions(overrides?)` -- for SDK client constructors
- `getAgentServerHttpClientOptions(overrides?)` -- for typed wrappers such as `RemoteEventsList`; application code must not import or construct the low-level `HttpClient`

These helpers read host, session API key, and working directory from the active backend
registry and env config, so callers never hardcode URLs or auth tokens.

```ts
// CORRECT
const data = await new ConversationClient(
  getAgentServerClientOptions(),
).getConversation(id);
const file = await new FileClient(
  getAgentServerClientOptions(),
).downloadTextFile(path);

// WRONG -- raw axios/fetch calls fail the no-direct-agent-server-calls.test.ts guard
const data = await axios.get(`${host}/api/conversations/${id}`);
const data = await fetch(`/api/conversations/${id}`);
```

**Allowed exceptions** (files that may use axios directly for infrastructure reasons):

- `src/api/automation-service/automation-service.api.ts`
- `src/api/cloud/proxy.ts` -- the proxy envelope POST itself
- `src/api/main-app-auth.ts` -- the local main-app authentication endpoint

### Rule 2 -- Cloud backend routes must go through `callCloudProxy`

Any call from the browser to the cloud backend (`app.all-hands.dev`) or a cloud
runtime sandbox (`*.prod-runtime.all-hands.dev`) **must** go through `callCloudProxy()`
in `src/api/cloud/proxy.ts`. These origins do not permit CORS from `localhost`;
`callCloudProxy` POSTs the request envelope to `/api/cloud-proxy` on the local
agent-server, which forwards it server-side.

```ts
import { callCloudProxy } from "../cloud/proxy";

// CORRECT -- cloud endpoint
const result = await callCloudProxy<ResponseType>({
  backend,
  method: "GET",
  path: `/api/v1/app-conversations/search?${params}`,
});

// CORRECT -- cloud runtime sandbox, auth via session key
const result = await callCloudProxy<ResponseType>({
  backend,
  method: "GET",
  hostOverride: buildHttpBaseUrl(conversationUrl),
  path: `/api/git/changes?path=${path}`,
  authMode: "session-api-key",
  sessionApiKey,
});

// WRONG -- direct fetch/axios to a cloud host is blocked by CORS in the browser
const result = await axios.get(`${backend.host}/api/v1/app-conversations`);
```

`callCloudProxy` key options:

- `backend` -- the cloud `Backend` object (provides host and bearer token)
- `hostOverride` -- override for runtime-sandbox calls; replaces `backend.host`
- `authMode` -- `"bearer"` (default, cloud) | `"session-api-key"` (runtime sandbox) | `"none"`
- `sessionApiKey` -- required when `authMode === "session-api-key"`

Standard cloud/local branch pattern used throughout the service layer:

```ts
if (getActiveBackend().backend.kind === "cloud") {
  return callCloudProxy({ backend: active, ... });
}
return new ConversationClient(getAgentServerClientOptions()).someMethod(...);
```
