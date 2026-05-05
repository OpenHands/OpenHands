# Plugin Launch Flow

This document traces the complete data flow for launching plugins in OpenHands, from the source marketplace through to agent execution. Each section shows the exact endpoints, payloads, and transformations.

## Architecture Overview

```
Marketplace ──▶ Plugin Directory ──▶ Frontend /launch ──▶ App Server ──▶ Agent Server ──▶ SDK
  (GitHub)        (Index + UI)          (Modal)            (API)        (in sandbox)    (plugin loading)
```

| Component | Responsibility |
|-----------|---------------|
| **Marketplace** | Source of truth for plugin catalog (GitHub repo) |
| **Plugin Directory** | Index plugins from marketplace, serve browsing UI, construct launch URLs |
| **Frontend** | Display confirmation modal, collect parameters, call API |
| **App Server** | Validate request, create conversation, pass plugin specs to agent server |
| **Agent Server** | Run inside sandbox, delegate plugin loading to SDK |
| **SDK** | Fetch plugins, load contents, merge skills/hooks/MCP into agent |

---

## Step 1: Marketplace (GitHub)

**Source**: `github.com/OpenHands/plugin-marketplace`

The marketplace is a GitHub repository containing a `catalog.yaml` that defines all available plugins.

### catalog.yaml

```yaml
name: "OpenHands Plugin Marketplace"
owner:
  name: "OpenHands"
  url: "https://github.com/OpenHands"
metadata:
  pluginRoot: "plugins"  # Optional: subdirectory containing plugins
plugins:
  - name: "city-weather"
    source: "github:jpshackelford/openhands-sample-plugins"
    ref: "main"
    repo_path: "plugins/city-weather"
    description: "Get current weather for any city"
    tags: ["weather", "utility"]
```

### Plugin Source (`plugin.json`)

Each plugin has a `plugin.json` in its `.claude-plugin/` directory:

```json
{
  "name": "city-weather",
  "description": "Get current weather for any city",
  "entry_command": "now",
  "parameters": {
    "city": {
      "type": "string",
      "description": "City name",
      "required": true,
      "default": "San Francisco"
    }
  },
  "examples": [
    {
      "title": "Check Tokyo weather",
      "prompt": "/city-weather:now Tokyo"
    }
  ]
}
```

**Output to Plugin Directory**: Raw catalog.yaml + individual plugin.json files

---

## Step 2: Plugin Directory Server

**Endpoints**:
- `GET /api/plugins` - List all plugins
- `GET /api/plugins/{id}` - Get plugin details
- `GET /api/plugins/{id}/config` - Get plugin config (entry_command, parameters, examples)

### GET /api/plugins

Fetches and transforms the marketplace catalog.

**Request**: None (fetches from configured `MARKETPLACE_SOURCE`)

**Response**:
```json
{
  "plugins": [
    {
      "id": "city-weather",
      "name": "city-weather",
      "description": "Get current weather for any city",
      "source": {
        "source": "github",
        "repo": "jpshackelford/openhands-sample-plugins",
        "ref": "main",
        "repo_path": "plugins/city-weather"
      },
      "tags": ["weather", "utility"]
    }
  ]
}
```

### GET /api/plugins/{id}/config

Fetches and returns the config fields from `plugin.json`.

**Request**: `GET /api/plugins/city-weather/config`

**Response** (200 OK):
```json
{
  "entry_command": "now",
  "parameters": {
    "city": {
      "type": "string",
      "description": "City name",
      "required": true,
      "default": "San Francisco"
    }
  },
  "examples": [
    {
      "title": "Check Tokyo weather",
      "prompt": "/city-weather:now Tokyo"
    }
  ]
}
```

**Output to Plugin Directory Client**: Plugin metadata + config

---

## Step 3: Plugin Directory Client

When user clicks "Launch", the client constructs a launch URL.

### buildLaunchUrl() Transformation

**Input**:
- Plugin: `{ name: "city-weather", source: { source: "github", repo: "...", ref: "main", repo_path: "plugins/city-weather" } }`
- Config: `{ entry_command: "now", parameters: { city: { type: "string", default: "San Francisco" } } }`

**Transformation**:
1. Build `PluginSpec` with source and parameter defaults
2. Base64-encode as `plugins` query param
3. Build slash command from `entry_command` as `message` query param

**Output** (Launch URL):
```
https://app.openhands.ai/launch?plugins=BASE64&message=/city-weather:now
```

Where `plugins` decodes to:
```json
[{
  "source": "github:jpshackelford/openhands-sample-plugins",
  "ref": "main",
  "repo_path": "plugins/city-weather",
  "parameters": {
    "city": "San Francisco"
  }
}]
```

**Note**: The `message` contains only the slash command (`/city-weather:now`), not parameter values. Parameter values are added by the launch modal when the user submits.

---

## Step 4: OpenHands Frontend (`/launch` Route)

**Route**: `/launch?plugins=BASE64&message=/city-weather:now`

[PR #12699](https://github.com/OpenHands/OpenHands/pull/12699)

### URL Parsing

**Input** (query params):
- `plugins`: Base64-encoded JSON array of PluginSpec
- `message`: Pre-filled slash command (optional)

**Decoded**:
```json
{
  "plugins": [{
    "source": "github:jpshackelford/openhands-sample-plugins",
    "ref": "main",
    "repo_path": "plugins/city-weather",
    "parameters": { "city": "San Francisco" }
  }],
  "message": "/city-weather:now"
}
```

### Modal Display

The frontend:
1. Displays plugin info and parameter form
2. Pre-fills parameter inputs with default values
3. Shows the message input pre-filled with `/city-weather:now`

### User Submits

User modifies parameters (e.g., changes city to "Tokyo") and clicks "Start Conversation".

**Output** (API call to App Server):

```
POST /api/v1/app-conversations
Content-Type: application/json
Authorization: Bearer <user_token>

{
  "plugins": [{
    "source": "github:jpshackelford/openhands-sample-plugins",
    "ref": "main",
    "repo_path": "plugins/city-weather",
    "parameters": {
      "city": "Tokyo"
    }
  }],
  "message": "/city-weather:now Tokyo"
}
```

**Note**: The message now includes the user's parameter value ("Tokyo").

---

## Step 5: OpenHands App Server

**Endpoint**: `POST /api/v1/app-conversations`

[PR #12338](https://github.com/OpenHands/OpenHands/pull/12338)

### Request Schema

```python
class PluginSpec(BaseModel):
    source: str                    # "github:owner/repo" or URL
    ref: str | None = None         # Git ref (branch/tag/commit)
    repo_path: str | None = None   # Subdirectory within repo
    parameters: dict | None = None # User-provided parameter values

class CreateAppConversationRequest(BaseModel):
    plugins: list[PluginSpec] | None = None
    message: str | None = None
    # ... other fields
```

### Processing

**Call stack**:
1. `AppConversationRouter.create_conversation()` receives request
2. `LiveStatusAppConversationService._finalize_conversation_request()`:
   - Converts `PluginSpec` → SDK `PluginSource` objects
   - Creates `initial_message` from `message` field
3. Creates `StartConversationRequest` for agent server

### Transformation

**Input** (API request):
```json
{
  "plugins": [{
    "source": "github:jpshackelford/openhands-sample-plugins",
    "ref": "main",
    "repo_path": "plugins/city-weather",
    "parameters": { "city": "Tokyo" }
  }],
  "message": "/city-weather:now Tokyo"
}
```

**Output** (to Agent Server):
```python
StartConversationRequest(
    plugins=[
        PluginSource(
            source="github:jpshackelford/openhands-sample-plugins",
            ref="main",
            repo_path="plugins/city-weather",
            parameters={"city": "Tokyo"}
        )
    ],
    initial_message=MessageParam(
        role="user",
        content=[{"type": "text", "text": "/city-weather:now Tokyo"}]
    ),
    # ... other fields
)
```

---

## Step 6: Agent Server (in Sandbox)

**Entry point**: `ConversationService.start_conversation()`

[SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651)

### Processing

**Call stack**:
1. `ConversationService.start_conversation(request)` receives `StartConversationRequest`
2. Creates `StoredConversation` with plugin specs persisted
3. Creates `LocalConversation(plugins=request.plugins, ...)`
4. Plugin loading deferred until first `run()` or `send_message()`

### Transformation

**Input** (`StartConversationRequest`):
```python
StartConversationRequest(
    plugins=[PluginSource(source="github:...", ref="main", repo_path="...", parameters={...})],
    initial_message=MessageParam(role="user", content=[...])
)
```

**Output** (`LocalConversation` created):
```python
LocalConversation(
    agent=agent,
    plugins=[PluginSource(...)],  # Stored, not yet loaded
    workspace=workspace,
    # ...
)
```

---

## Step 7: SDK Plugin Loading

**Trigger**: First `conversation.run()` or `conversation.send_message()`

[SDK PR #1647](https://github.com/OpenHands/software-agent-sdk/pull/1647)

### Processing

**Call stack**:
1. `LocalConversation._ensure_plugins_loaded()` triggered
2. For each `PluginSource`:
   - `Plugin.fetch(source, ref, repo_path)` → clones/caches git repo
   - `Plugin.load(path)` → parses `plugin.json`, loads commands/skills/hooks
3. `plugin.add_skills_to(skill_context)` → merges skills into agent
4. `plugin.add_mcp_config_to(mcp_config)` → merges MCP servers

### Plugin.fetch() Transformation

**Input** (`PluginSource`):
```python
PluginSource(
    source="github:jpshackelford/openhands-sample-plugins",
    ref="main",
    repo_path="plugins/city-weather",
    parameters={"city": "Tokyo"}
)
```

**Output** (`Plugin` object):
```python
Plugin(
    name="city-weather",
    path="/tmp/plugins/city-weather",
    manifest=PluginManifest(
        name="city-weather",
        entry_command="now",
        commands={"now": Command(...)},
        skills=[Skill(...)],
        hooks={...},
        mcp_servers={...}
    ),
    parameters={"city": "Tokyo"}  # Passed through for skill interpolation
)
```

---

## Step 8: Agent Receives Message

The agent now has:
- Plugin skills merged into its skill context
- MCP servers configured and running
- The initial message `/city-weather:now Tokyo` in its conversation

When the agent processes the message:
1. Recognizes `/city-weather:now` as a slash command
2. Looks up the `now` command from the `city-weather` plugin
3. Executes the command with parameter `city=Tokyo`

---

## Complete Data Flow Summary

| Step | Component | Input | Output |
|------|-----------|-------|--------|
| 1 | Marketplace | - | `catalog.yaml` + `plugin.json` files |
| 2 | Plugin Directory Server | Marketplace files | REST API responses |
| 3 | Plugin Directory Client | Plugin + Config | Launch URL with base64 `plugins` + `message` |
| 4 | OpenHands Frontend | URL query params | `POST /api/v1/app-conversations` |
| 5 | App Server | API request | `StartConversationRequest` to agent server |
| 6 | Agent Server | `StartConversationRequest` | `LocalConversation` with deferred plugins |
| 7 | SDK | `PluginSource` list | Loaded `Plugin` objects with skills/hooks/MCP |
| 8 | Agent | Initial message | Command execution |

---

## Key Design Decisions

### Plugin Loading in Sandbox

Plugins load **inside the sandbox** because:
- Plugin hooks and scripts need isolated execution
- MCP servers run inside the sandbox
- Skills may reference sandbox filesystem

### Entry Command vs Full Message

The `entry_command` field contains only the command name (e.g., `"now"`), not the full slash command. This separation allows:
- Plugin Directory to construct the slash command from plugin name + entry_command
- Launch modal to append user-provided parameter values
- Flexibility for the launch experience to differ from direct SDK usage

### Parameter Flow

Parameters flow through two paths:
1. **In URL/API**: As structured data in `PluginSpec.parameters` for validation and form rendering
2. **In message**: As text appended to the slash command for the agent to parse

---

## Related PRs

- [OpenHands PR #12338](https://github.com/OpenHands/OpenHands/pull/12338) - App server plugin support
- [OpenHands PR #12699](https://github.com/OpenHands/OpenHands/pull/12699) - Frontend `/launch` route
- [SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651) - Agent server plugin loading
- [SDK PR #1647](https://github.com/OpenHands/software-agent-sdk/pull/1647) - Plugin.fetch() for remote plugin fetching
- [SDK PR #2230](https://github.com/OpenHands/software-agent-sdk/pull/2230) - entry_command field definition
- [Plugin Directory PR #84](https://github.com/OpenHands/plugin-directory/pull/84) - entry_command support in plugin directory
