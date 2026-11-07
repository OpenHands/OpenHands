# An agent drives a session through the Canvas proxy, with only the session key

The DigitalOcean tunnel is the one part stood in: session `sess_e2e` is
"connected" to a local agent-server started with no key, the way the guest
in a DigitalOcean sandbox runs. Everything else is real: the Canvas
`/mars` routes (`proxy.mjs`, using `scripts/mars-web-bridge.mjs`) and the
OpenHands Python SDK 1.56.0 (`agent.py`, with `RemoteWorkspace` and
`Conversation`).

```sh
agent-server --host 127.0.0.1 --port 18850            # no SESSION_API_KEY
GUEST_PORT=18850 PROXY_PORT=18860 CANVAS_KEY=agent-e2e-key node proxy.mjs
PROXY_PORT=18860 CANVAS_KEY=agent-e2e-key GUEST_DIR=... python agent.py
```

## After (this PR)

```
wrong key refused: exit -1 | Remote execution error: Client error '401 Unauthorized' for url 'http://127.0.0.1:18860/ma
terminal: 0 hello-from-the-session | root
conversation: 2788f048-38fe-4400-b522-a3218c5dbdf1 | events so far: ['ConversationStateUpdateEvent']
done
```

What the proxy saw:

```
HTTP POST /mars/sessions/sess_e2e/api/bash/start_bash_command key=yes
HTTP POST /mars/sessions/sess_e2e/api/bash/start_bash_command key=yes
HTTP GET /mars/sessions/sess_e2e/api/bash/bash_events/search?kind__eq=BashOutput&sort_order=TIMESTAMP&limit=100&command_id__eq=fc27a0a877194a9b8e8be62e7b559eaa key=yes
HTTP GET /mars/sessions/sess_e2e/api/bash/bash_events/search?kind__eq=BashOutput&sort_order=TIMESTAMP&limit=100&command_id__eq=fc27a0a877194a9b8e8be62e7b559eaa key=yes
HTTP POST /mars/sessions/sess_e2e/api/conversations key=yes
UPGRADE /mars/sessions/sess_e2e/sockets/events/2788f048-38fe-4400-b522-a3218c5dbdf1 key-header=no cookie=no
```

The event socket upgrade carries no header and no cookie; it logs in with
its first message, and the SDK's conversation becomes ready.

## Before (#18706, the PR below this one)

Same agent, same agent-server, the proxy from the previous PR:

```
wrong key refused: exit -1 | Remote execution error: Client error '401 Unauthorized' for url 'http://127.0.0.1:18861/ma
terminal: 0 hello-from-the-session | root
openhands.sdk.conversation.exceptions.WebSocketConnectionError: WebSocket subscription did not complete within 30.0 seconds for conversation edc956f5-4644-409e-9f11-359959ca3d6c. Events may be missed.
```

REST worked, but the event socket was refused at the upgrade (401), so the
SDK gave up on the conversation.

## In the Canvas page too (20 Oct)

The same session path, driven from the browser with `control-openhands`. It
checks that the page still gets a prepared sandbox now that the browser no
longer prepares it. DigitalOcean's API and tunnel are stood in by
`standin-do.mjs` (one OpenHands agent, one ready session, tunnel piped to a
local agent-server). Canvas, its bridge and the agent-server are real.

```sh
agent-server --host 127.0.0.1 --port 18931     # no session key; given its own OH_SECRET_KEY
FAKE_PORT=18930 GUEST_PORT=18931 node standin-do.mjs
# the run's launcher env: MARS_API_BASE_URL=http://127.0.0.1:18930
```

1. Add backend → Providers → DigitalOcean, signed in with the placeholder
   token (`dop_v1_` + 64 zeros). The stand-in's agent and session listed.
2. Connected `stand-in-session`. The tunnel opened (`TUNNEL
   /v2/agents/sessions/sess_standin/port-forward/8000`), and the host
   prepared the sandbox: the guest's `agent_settings.tools` went from `null`
   to the subprocess terminal.
3. `control-openhands restart`, then a reload. The session reconnected from
   the saved sign-in, and a fresh guest was prepared the same way.
4. With an LLM set on the guest, a chat asked for
   `echo hello-from-the-session && uname -s`. The agent ran it in the
   session's terminal: `hello-from-the-session`, `Linux`
   (`session-conversation.png`).

The secret-key half of the preparation reports `unsupported` here, since a
local agent-server is not the guest's `openhands-agent-server` under s6. The
stand-in guest was started with its own key instead. Without one, its
settings answered 503 once an LLM key was stored: the failure that half
works around.
