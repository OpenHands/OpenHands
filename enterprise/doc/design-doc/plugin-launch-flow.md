# Plugin Launch Flow

This document describes how plugins are launched in OpenHands Saas / Enterprise, from the plugin directory through to agent execution.

## Architecture Overview

```
Plugin Directory ──▶ Frontend /launch ──▶ App Server ──▶ Agent Server ──▶ SDK
    (external)           (modal)           (API)        (in sandbox)    (plugin loading)
```

| Component | Responsibility |
|-----------|---------------|
| **Plugin Directory** | Index plugins, present to user, construct launch URLs |
| **Frontend** | Display confirmation modal, collect parameters, call API |
| **App Server** | Validate request, pass plugin specs to agent server |
| **Agent Server** | Run inside sandbox, delegate plugin loading to SDK |
| **SDK** | Fetch plugins, load contents, merge skills/hooks/MCP into agent |

## User Experience

### Plugin Directory

The plugin directory presents users with a catalog of available plugins. For each plugin, users see:
- Plugin name and description (from `plugin.json`)
- Author and version information
- A "Launch" button

When a user clicks "Launch", the plugin directory:
1. Reads the plugin's `entry_command` to know which slash command to invoke
2. Determines what parameters the plugin accepts (if any)
3. Redirects to OpenHands with this information encoded in the URL

### Parameter Collection

If a plugin requires user input (API keys, configuration values, etc.), the frontend displays a form modal before starting the conversation. Parameters are passed in the launch URL and rendered as form fields based on their type:

- **String values** → Text input
- **Number values** → Number input
- **Boolean values** → Checkbox

Only primitive types are supported. Complex types (arrays, objects) are not currently supported for parameter input.

The user fills in required values, then clicks "Start Conversation" to proceed.

## Launch Flow

1. **Plugin Directory** (external) constructs a launch URL to the OpenHands app server when user clicks "Launch":
   ```
   /launch?plugins=BASE64_JSON&message=/city-weather:now%20Tokyo
   ```

   The `plugins` parameter includes any parameter definitions with default values:
   ```json
   [{
     "source": "github:owner/repo",
     "repo_path": "plugins/my-plugin",
     "parameters": {"api_key": "", "timeout": 30, "debug": false}
   }]
   ```

2. **OpenHands Frontend** (`/launch` route, [PR #12699](https://github.com/OpenHands/OpenHands/pull/12699)) displays modal with parameter form, collects user input

3. **OpenHands App Server** ([PR #12338](https://github.com/OpenHands/OpenHands/pull/12338)) receives the API call:
   ```
   POST /api/v1/app-conversations
   {
     "plugins": [{"source": "github:owner/repo", "repo_path": "plugins/city-weather"}],
     "initial_message": {"content": [{"type": "text", "text": "/city-weather:now Tokyo"}]}
   }
   ```

   Call stack:
   - `AppConversationRouter` receives request with `PluginSpec` list
   - `LiveStatusAppConversationService._finalize_conversation_request()` converts `PluginSpec` → `PluginSource`
   - Creates `StartConversationRequest(plugins=sdk_plugins, ...)` and sends to agent server

4. **Agent Server** (inside sandbox, [SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651)) stores specs, defers loading:

   Call stack:
   - `ConversationService.start_conversation()` receives `StartConversationRequest`
   - Creates `StoredConversation` with plugin specs
   - Creates `LocalConversation(plugins=request.plugins, ...)`
   - Plugin loading deferred until first `run()` or `send_message()`

5. **SDK** fetches and loads plugins on first use:

   Call stack:
   - `LocalConversation._ensure_plugins_loaded()` triggered by first message
   - For each plugin spec:
     - `Plugin.fetch(source, ref, repo_path)` → clones/caches git repo
     - `Plugin.load(path)` → parses `plugin.json`, loads commands/skills/hooks
     - `plugin.add_skills_to(context)` → merges skills into agent
     - `plugin.add_mcp_config_to(config)` → merges MCP servers

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

- [OpenHands PR #12338](https://github.com/OpenHands/OpenHands/pull/12338) - App server plugin support
- [OpenHands PR #12699](https://github.com/OpenHands/OpenHands/pull/12699) - Frontend `/launch` route
- [SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651) - Agent server plugin loading
- [SDK PR #1647](https://github.com/OpenHands/software-agent-sdk/pull/1647) - Plugin.fetch() for remote plugin fetching
