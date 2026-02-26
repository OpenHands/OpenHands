# Plugin Launch Flow

This document describes how plugins are launched in OpenHands, from the plugin directory through to agent execution.

## Architecture Overview

```
Plugin Directory ──▶ Frontend /launch ──▶ App Server ──▶ Agent Server ──▶ SDK
    (external)           (modal)           (API)        (in sandbox)    (plugin loading)
```

| Component | Responsibility |
|-----------|---------------|
| **Plugin Directory** | Index plugins, construct launch URLs with entry commands |
| **Frontend** | Display confirmation modal, collect parameters, call API |
| **App Server** | Validate request, pass plugin specs to agent server |
| **Agent Server** | Run inside sandbox, delegate plugin loading to SDK |
| **SDK** | Fetch plugins, load contents, merge skills/hooks/MCP into agent |

## Launch Flow

1. **Plugin Directory** constructs a launch URL when user clicks "Launch":
   ```
   /launch?plugins=BASE64_JSON&message=/city-weather:now%20Tokyo
   ```

2. **Frontend** (`/launch` route) displays modal, collects any parameters, then calls:
   ```
   POST /api/v1/app-conversations
   {
     "plugins": [{"source": "github:owner/repo", "repo_path": "plugins/city-weather"}],
     "initial_message": {"content": [{"type": "text", "text": "/city-weather:now Tokyo"}]}
   }
   ```

3. **App Server** passes plugin specs to agent server via `StartConversationRequest`

4. **Agent Server** (inside sandbox) stores specs, defers loading to first message

5. **SDK** fetches plugin repo, loads `plugin.json`, merges skills/hooks/MCP into agent

6. **Agent** receives message, `/city-weather:now` triggers the skill

## Key Design Decisions

### Plugin Loading in Sandbox

Plugins load **inside the sandbox** because:
- Plugin hooks and scripts need isolated execution
- MCP servers run inside the sandbox
- Skills may reference sandbox filesystem

### Entry Command Handling

The `entry_command` field in `plugin.json` allows plugin authors to declare a default command:

```json
{
  "name": "city-weather",
  "entry_command": "now"
}
```

This flows through the system:
1. Plugin author declares `entry_command` in plugin.json
2. Plugin directory reads it when indexing
3. Plugin directory includes `/city-weather:now` in the launch URL's `message` parameter
4. Message passes through to agent as `initial_message`

The SDK exposes this field but does not auto-invoke it—callers control the initial message.

## Related

- [PR #12338](https://github.com/OpenHands/OpenHands/pull/12338) - App server plugin support
- [PR #12699](https://github.com/OpenHands/OpenHands/pull/12699) - Frontend `/launch` route
- [SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651) - Agent server plugin loading
