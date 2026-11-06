# Let an agent use your DigitalOcean backend

People sign in to DigitalOcean in Canvas; agents use the Canvas session key.
Once you are signed in, an agent that holds that key can list your
DigitalOcean agents, start or wake a session, and run a conversation in it,
on the team that is active in Canvas.

## The key

- An agent in a conversation on Canvas's Local backend (the Agent Server
  that Canvas starts on this machine or in its Docker container) already
  has it, as `$OPENHANDS_AUTOMATION_API_KEY`. The launcher sets that for
  Automations; it is the same key. (A Docker deployment that sets
  `MARS_WEB_KEY` makes `/mars/*` want that key instead.) Nothing in the
  agent's prompt mentions DigitalOcean, so point it at this page.
- On the machine that runs Canvas, it is in
  `~/.openhands/agent-canvas/api-key.txt` (Docker: `$STATE_DIR/api-key.txt`,
  or whatever `LOCAL_BACKEND_API_KEY` is set to).

**What the key can do.** Whoever holds it can do anything Canvas can,
DigitalOcean included: start sessions (they bill while they run), stop them,
and delete agents, for the active team. Give it only to agents you would let
do that.

The agent must be able to reach Canvas. By default Canvas listens on
`127.0.0.1`, so that means an agent on the same machine. An agent inside a
DigitalOcean session, in the cloud or in CI cannot reach it; it should use its
own DigitalOcean token with DigitalOcean directly
(`doctl agents port-forward`).

These calls are served by the browser build of Canvas (the `npx` launcher
or Docker), and only when it is started with `MARS_WEB=1`: the DigitalOcean
integration is still experimental, so it is off by default. The desktop app
keeps its DigitalOcean bridge inside the app and doesn't serve them yet.

## The calls

Every call is `POST /mars/rpc/<method>` with the arguments as
`{"args": [...]}` and the key in `X-Session-API-Key`.

```sh
B="http://localhost:8000"   # the Canvas address, as this agent reaches it
KEY="$OPENHANDS_AUTOMATION_API_KEY"
rpc() {
  curl -s -X POST "$B/mars/rpc/$1" -H "X-Session-API-Key: $KEY" \
    -H 'Content-Type: application/json' -d "{\"args\":$2}"
}

rpc listAgentConfigs '[]'                 # the team's agents; use "agent":"openhands"
rpc listConfigSessions '["<agent id>"]'   # its sessions
rpc createSession '["<agent id>", "my-task"]'   # or start one: bills while it runs
rpc openTunnel '[{"sessionId":"<session id>"}]' # connects (wakes a paused one)
rpc pauseSession '["<session id>"]'       # when done
```

`openTunnel` answers with a `host` such as
`$B/mars/sessions/<session id>`. That is the session's Agent Server: use its
REST API under that prefix with the same `X-Session-API-Key` header. Canvas
prepares the sandbox before `openTunnel` answers and says how that went in
`sandbox`: `prepared` means it is ready to use; `unreachable` means its Agent
Server never answered (`upstreamFailure` may say why); `failed` means a
preparation step failed, so the terminal and encrypted settings may not work
until the next `openTunnel` tries again.

## With the OpenHands SDK

Point a remote workspace at that host. The SDK sends the key on REST calls
and as the first message on the event socket, which Canvas accepts:

```python
from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.workspace import RemoteWorkspace

workspace = RemoteWorkspace(host=host, api_key=key, working_dir="/workspace/project")
print(workspace.execute_command("git status").stdout)

agent = Agent(llm=LLM(model="...", api_key="..."), tools=[...])
conversation = Conversation(agent=agent, workspace=workspace)
conversation.send_message("Fix the failing test")
conversation.run()
```

The conversation needs an LLM: send one in the request, as above, or set it
in that session's Agent Server settings.

## Event sockets

`/mars/sessions/<id>/sockets/...` accepts the key in any of three ways: the
`X-Session-API-Key` header, the Canvas cookie, or (what the SDK and the
TypeScript client do) a first message `{"type":"auth","session_api_key":"..."}`
sent within 10 seconds. A wrong header or a stale cookie is refused with 401
at once; only a socket that brings neither may log in by its first message.
Canvas does not connect to the session until the key checks out; a wrong or
missing first-message key closes the socket with 4001, as the Agent Server
does. The guest's pings reach the agent, so a dead connection is noticed.
