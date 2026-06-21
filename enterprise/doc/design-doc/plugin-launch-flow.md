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

**Source**: A GitHub repository (e.g., `github.com/OpenHands/plugin-marketplace`)

The marketplace is a GitHub repository containing a `marketplace.json` that indexes all available plugins.

### marketplace.json

```json
{
  "name": "OpenHands Plugin Marketplace",
  "owner": {
    "name": "OpenHands",
    "email": "team@all-hands.dev"
  },
  "metadata": {
    "description": "Official OpenHands plugin marketplace",
    "pluginRoot": "plugins"
  },
  "plugins": [
    {
      "name": "city-weather",
      "source": "github:jpshackelford/openhands-sample-plugins",
      "ref": "main",
      "repo_path": "plugins/city-weather",
      "description": "Get current weather for any city",
      "tags": ["weather", "utility"]
    }
  ]
}
```

### Plugin Source (`plugin.json`)

Each plugin has a `plugin.json` in its `.claude-plugin/` directory. This file contains both official plugin manifest fields and optional directory-specific config fields:

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

**Output to Plugin Directory**: `marketplace.json` + individual `plugin.json` files

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

When user clicks "Launch", the client constructs a launch URL using `buildLaunchUrl()`.

### buildLaunchUrl() Input

From Plugin Directory Server APIs:
- **Plugin** (from `/api/plugins/{id}`):
  ```json
  {
    "name": "city-weather",
    "source": {
      "source": "github",
      "repo": "jpshackelford/openhands-sample-plugins",
      "ref": "main",
      "repo_path": "plugins/city-weather"
    }
  }
  ```
- **PluginConfig** (from `/api/plugins/{id}/config`):
  ```json
  {
    "entry_command": "now",
    "parameters": {
      "city": { "type": "string", "required": true, "default": "San Francisco" }
    }
  }
  ```

### buildLaunchUrl() Transformation

1. **Build PluginSpec** from plugin source:
   - `source`: Convert to string format `"github:owner/repo"`
   - `ref`: Extract git ref if present
   - `repo_path`: Extract subdirectory path if present
   - `parameters`: Extract default values from parameter definitions

2. **Build message** using `buildEntrySlashCommand(pluginName, entryCommand)`:
   - Combines plugin name + entry_command → `"/city-weather:now"`
   - Does NOT include parameter values (App Server will add them later)

3. **Encode and construct URL**:
   - Base64-encode the PluginSpec array as `plugins` query param
   - Add slash command as `message` query param

### buildLaunchUrl() Output

**Launch URL**:
```
https://app.openhands.ai/launch?plugins=BASE64&message=%2Fcity-weather%3Anow
```

Where `plugins` (base64-decoded) contains:
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

And `message` (URL-decoded) is:
```
/city-weather:now
```

**Key point**: The `parameters` in the PluginSpec contain **default values** for pre-filling the launch modal form. The `message` contains only the slash command—the Frontend passes it through unchanged, and the App Server appends the parameter values as a formatted text block.

---

## Step 4: OpenHands Frontend (`/launch` Route)

**Route**: `/launch?plugins=BASE64&message=/city-weather:now`

[PR #12699](https://github.com/OpenHands/OpenHands/pull/12699)

### Input (from URL query params)

- `plugins`: Base64-encoded JSON array of PluginSpec
- `message`: Pre-filled slash command (no parameter values)

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

The frontend displays a confirmation modal:
1. Shows plugin info
2. Renders parameter form fields based on `plugins[].parameters`:
   - Text input for `city`, pre-filled with `"San Francisco"`
3. Shows message preview: `/city-weather:now`

### User Submits

When user clicks "Start Conversation":

1. **Collect final parameter values** from form inputs:
   - User changed `city` from `"San Francisco"` to `"Tokyo"`

2. **Update PluginSpec parameters** with user's values:
   ```json
   "parameters": { "city": "Tokyo" }
   ```

3. **Pass message through unchanged**:
   - The message `/city-weather:now` is NOT modified by the Frontend
   - Parameter values are passed in `plugins[].parameters`, not in the message

### Output (API call to App Server)

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
  "initial_message": {
    "role": "user",
    "content": [{"type": "text", "text": "/city-weather:now"}]
  }
}
```

**Summary of transformations**:
| Field | Input (from URL) | Output (to API) |
|-------|------------------|-----------------|
| `plugins[].parameters` | Default values (`"San Francisco"`) | User's values (`"Tokyo"`) |
| `initial_message.text` | Slash command (`/city-weather:now`) | Slash command unchanged (`/city-weather:now`) |

**Note**: The Frontend does NOT append parameter values to the message. Parameters are passed as structured data in `plugins[].parameters`. The App Server will append them to the message text (see Step 5).

---

## Step 5: OpenHands App Server

**Endpoint**: `POST /api/v1/app-conversations`

[PR #12338](https://github.com/OpenHands/OpenHands/pull/12338)

### Input (API Request)

```json
{
  "plugins": [{
    "source": "github:jpshackelford/openhands-sample-plugins",
    "ref": "main",
    "repo_path": "plugins/city-weather",
    "parameters": { "city": "Tokyo" }
  }],
  "initial_message": {
    "role": "user",
    "content": [{"type": "text", "text": "/city-weather:now"}]
  }
}
```

Note: The `initial_message.text` contains only the slash command—parameter values come separately in `plugins[].parameters`.

### Request Schema

```python
class PluginSpec(PluginSource):
    """Extends SDK's PluginSource with user-provided parameters."""
    parameters: dict[str, Any] | None = None  # User-provided values

class AppConversationStartRequest(BaseModel):
    plugins: list[PluginSpec] | None = None
    initial_message: SendMessageRequest | None = None
    # ... other fields
```

### Processing & Transformation

**Call stack** in `LiveStatusAppConversationService`:

1. **`_construct_initial_message_with_plugin_params()`** - Appends parameters to message:
   ```python
   # Original message: "/city-weather:now"
   # Parameters: {"city": "Tokyo"}
   # Result: "/city-weather:now\n\nPlugin Configuration Parameters:\n- city: Tokyo"
   ```

2. **Convert PluginSpec → SDK PluginSource** (parameters are DROPPED):
   ```python
   sdk_plugins = [
       PluginSource(
           source=p.source,      # "github:jpshackelford/openhands-sample-plugins"
           ref=p.ref,            # "main"
           repo_path=p.repo_path # "plugins/city-weather"
       )
       # NOTE: p.parameters is NOT passed to SDK PluginSource!
       for p in plugins
   ]
   ```

3. **Create StartConversationRequest** for agent server

### Output (to Agent Server)

```python
StartConversationRequest(
    plugins=[
        PluginSource(
            source="github:jpshackelford/openhands-sample-plugins",
            ref="main",
            repo_path="plugins/city-weather"
            # NO parameters field - SDK PluginSource doesn't have it
        )
    ],
    initial_message=SendMessageRequest(
        content=[
            TextContent(
                text="/city-weather:now\n\nPlugin Configuration Parameters:\n- city: Tokyo"
            )
        ]
    ),
    # ... other fields
)
```

**⚠️ CRITICAL**: Plugin parameters are passed to the agent via **message text**, not via the `PluginSource` object. The SDK's `PluginSource` class only has `source`, `ref`, and `repo_path` fields.

**Note on message construction**: The original slash command `/city-weather:now` does NOT include the parameter value "Tokyo" inline. The parameter appears only in the formatted "Plugin Configuration Parameters" block appended by the App Server.

---

## Step 6: Agent Server (in Sandbox)

**Entry point**: `ConversationService.start_conversation()`

[SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651)

### Input (`StartConversationRequest`)

```python
StartConversationRequest(
    plugins=[
        PluginSource(
            source="github:jpshackelford/openhands-sample-plugins",
            ref="main",
            repo_path="plugins/city-weather"
        )
    ],
    initial_message=SendMessageRequest(
        content=[
            TextContent(
                text="/city-weather:now\n\nPlugin Configuration Parameters:\n- city: Tokyo"
            )
        ]
    )
)
```

### Processing

**Call stack**:
1. `ConversationService.start_conversation(request)` receives `StartConversationRequest`
2. Creates `StoredConversation` with plugin specs persisted
3. Creates `LocalConversation(plugins=request.plugins, ...)`
4. Plugin loading deferred until first `run()` or `send_message()`

### Output (`LocalConversation`)

```python
LocalConversation(
    agent=agent,
    plugins=[PluginSource(...)],  # Stored, not yet loaded
    workspace=workspace,
    # initial_message queued for processing
)
```

---

## Step 7: SDK Plugin Loading

**Trigger**: First `conversation.run()` or `conversation.send_message()`

[SDK PR #1647](https://github.com/OpenHands/software-agent-sdk/pull/1647)

### Input (`PluginSource` list)

```python
[
    PluginSource(
        source="github:jpshackelford/openhands-sample-plugins",
        ref="main",
        repo_path="plugins/city-weather"
    )
]
```

### Processing

**Call stack**:
1. `LocalConversation._ensure_plugins_loaded()` triggered
2. For each `PluginSource`:
   - `Plugin.fetch(source, ref, repo_path)` → clones/caches git repo
   - `Plugin.load(path)` → parses `plugin.json`, loads commands/skills/hooks
3. `plugin.add_skills_to(skill_context)` → merges skills into agent
4. `plugin.add_mcp_config_to(mcp_config)` → merges MCP servers

### Output (`Plugin` object)

```python
Plugin(
    name="city-weather",
    path="/tmp/plugins/city-weather",
    manifest=PluginManifest(
        name="city-weather",
        entry_command="now",       # Read from plugin.json
        commands={"now": Command(...)},
        skills=[Skill(...)],
        hooks={...},
        mcp_servers={...}
    )
    # NOTE: No parameters field - parameters are in the message text
)
```

---

## Step 8: Agent Receives Message

The agent now has:
- Plugin skills merged into its skill context (including `/city-weather:now` command as a skill)
- MCP servers configured and running
- The initial message in its conversation

### Message Content

```
/city-weather:now

Plugin Configuration Parameters:
- city: Tokyo
```

### Processing

When the agent processes the message:
1. Recognizes `/city-weather:now` as a slash command (keyword trigger)
2. The `KeywordTrigger` activates the command skill
3. The agent reads the parameter value from the "Plugin Configuration Parameters" block
4. The skill executes with `city=Tokyo`

**Note**: Parameters are NOT passed as structured data to the plugin. The agent reads them from the message text in the formatted "Plugin Configuration Parameters" block appended by the App Server.

---

## Complete Data Flow Summary

| Step | Component | Input | Output |
|------|-----------|-------|--------|
| 1 | Marketplace | - | `marketplace.json` + `plugin.json` files |
| 2 | Plugin Directory Server | Marketplace files | REST API responses with `entry_command`, `parameters` |
| 3 | Plugin Directory Client | Plugin + Config | Launch URL: `plugins` (with defaults) + `message` (slash command only) |
| 4 | OpenHands Frontend | URL query params | API call: `plugins` (with user values) + `message` (unchanged slash command) |
| 5 | App Server | API request | `StartConversationRequest`: `PluginSource` (no params) + message (params in text) |
| 6 | Agent Server | `StartConversationRequest` | `LocalConversation` with deferred plugins |
| 7 | SDK | `PluginSource` list | Loaded `Plugin` objects with skills/hooks/MCP |
| 8 | Agent | Initial message with params in text | Command execution |

### Parameter Journey

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Plugin Directory   │     │  OpenHands Frontend │     │    App Server       │
│                     │     │                     │     │                     │
│  plugins[].params   │────▶│  plugins[].params   │────▶│  Appends params to  │
│  = defaults         │     │  = user values      │     │  message as text    │
│                     │     │  (from form edit)   │     │  block, then DROPS  │
│                     │     │                     │     │  from PluginSource  │
│  message =          │     │  message =          │     │                     │
│  /cmd:entry         │────▶│  /cmd:entry         │────▶│  Final message:     │
│  (no values)        │     │  (unchanged!)       │     │  /cmd:entry         │
│                     │     │                     │     │  + params block     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

**Key insight**: The Frontend does NOT modify the message. It passes the slash command through unchanged and sends parameters as structured data in `plugins[].parameters`. The App Server is responsible for formatting parameters into the message text.

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
- Frontend to collect user-provided parameter values via form UI
- App Server to format parameters into the message text
- Flexibility for the launch experience to differ from direct SDK usage

### Parameter Flow (Important!)

Parameters travel through the system as **structured data until the App Server**, where they are converted to text:

1. **Structured data path** (`PluginSpec.parameters`):
   - Plugin Directory → Frontend → App Server API
   - Used for form rendering (pre-fill defaults, collect user edits)
   - **Formatted into message text by App Server**
   - **Then dropped** (not passed to SDK `PluginSource`)

2. **Message path**:
   - Plugin Directory sends slash command only (e.g., `/city-weather:now`)
   - Frontend passes it through **unchanged**
   - App Server appends formatted parameter block to the message
   - **This is how the agent receives parameter values**

The SDK's `PluginSource` class intentionally does NOT have a `parameters` field. All parameter context is communicated to the agent via the initial message text, specifically in the "Plugin Configuration Parameters" block appended by the App Server.

---

## Related PRs

- [OpenHands PR #12338](https://github.com/OpenHands/OpenHands/pull/12338) - App server plugin support
- [OpenHands PR #12699](https://github.com/OpenHands/OpenHands/pull/12699) - Frontend `/launch` route
- [SDK PR #1651](https://github.com/OpenHands/software-agent-sdk/pull/1651) - Agent server plugin loading
- [SDK PR #1647](https://github.com/OpenHands/software-agent-sdk/pull/1647) - Plugin.fetch() for remote plugin fetching
- [SDK PR #2230](https://github.com/OpenHands/software-agent-sdk/pull/2230) - entry_command field definition
- [Plugin Directory PR #84](https://github.com/OpenHands/plugin-directory/pull/84) - entry_command support in plugin directory
