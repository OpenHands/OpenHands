# OpenHands Agent Execution Lifecycle - Key Entry Points

## Overview
This document maps the critical entry points in the OpenHands agent execution lifecycle, showing where agent steps are executed, LLM calls occur, tools are registered, and events are published to clients.

---

## 1. AGENT STEP FUNCTION / RUN LOOP

### 1.1 Conversation Start Orchestration
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/app_conversation_router.py`
**Lines:** 360-406

**Function:** `async def start_app_conversation()`
- **Purpose:** Entry point for starting a new sandboxed conversation
- **Key Actions:**
  1. Calls `app_conversation_service.start_app_conversation(start_request)`
  2. Returns initial `AppConversationStartTask` immediately
  3. Spawns background task via `asyncio.create_task(_consume_remaining)` to complete setup
  4. Tracks conversation creation analytics

**Related Class:** `AppConversationService` (Abstract base)
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/app_conversation_service.py`
**Lines:** 21-167

### 1.2 Main Conversation Start Implementation
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 232-410

**Method:** `async def start_app_conversation()` and `async def _start_app_conversation()`
**Execution Flow:**
1. **Lines 232-239:** Async generator wrapper that saves start task status updates
2. **Lines 241-410:** Main implementation performs sequential setup:
   - Lines 269-270: Wait for sandbox to start (calls `_wait_for_sandbox_start`)
   - Lines 294-302: Run setup scripts (clones repo, installs hooks, loads skills)
   - Lines 305-320: Build `StartConversationRequest` with LLM, MCP, tools, secrets
   - Lines 328-343: POST to agent-server `/api/conversations` endpoint
   - Lines 364-382: Save conversation info to database
   - Lines 384-400: Set up event callbacks (processors)
   - Lines 401-410: Process pending messages that queued before startup

**Key Data Flow:**
```
start_app_conversation()
  → _wait_for_sandbox_start()
  → run_setup_scripts()
  → _build_start_conversation_request_for_user()
      (configures LLM, tools, secrets, MCP)
  → httpx POST to agent-server /api/conversations
  → save_app_conversation_info()
  → _process_pending_messages()
```

### 1.3 Agent Context Building
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1214-1413

**Method:** `async def _build_start_conversation_request_for_user()`
**Purpose:** Constructs complete agent configuration before sending to agent server

**Key Configuration Steps (Lines 1265-1400):**
1. **Secrets Setup (Lines 1268-1289):**
   - Loads secrets from git providers and database
   - Merges with API-provided secrets
   - Creates `StaticSecret` and `LookupSecret` objects

2. **LLM + MCP Configuration (Lines 1291-1294):**
   - Calls `_configure_llm_and_mcp()`
   - Returns configured LLM instance and MCP config dict

3. **System Message Setup (Lines 1296-1313):**
   - Planning agent gets special instruction (`PLANNING_AGENT_INSTRUCTION`)
   - Adds web host context if available

4. **Tools Registration (Lines 1315-1329):**
   - For PLAN agent: `get_planning_tools(plan_path=...)`
   - For DEFAULT agent: `get_default_tools(enable_browser=True, ...)`
   - Registers built-in agents if sub-agents enabled
   - Gets agent definitions list

5. **Agent Creation (Lines 1331-1349):**
   - Creates `AgentSettings` with resolved tools, LLM, MCP
   - Calls `create_agent()` method
   - Applies server-side overrides (system prompts, LLM metadata)

6. **Hooks Loading (Lines 1350-1368):**
   - Loads hooks from remote workspace via `_load_hooks_from_workspace()`
   - Returns `HookConfig` object

7. **Plugins Incorporation (Lines 1370-1383):**
   - Constructs initial message with plugin parameters
   - Creates `PluginSource` list for SDK

8. **Request Building (Lines 1385-1400):**
   - Creates `ConversationSettings` with all components
   - Delegates to SDK's `create_request()` method

9. **Skills Loading (Lines 1402-1411):**
   - Loads workspace skills via `_load_skills_onto_request()`
   - Merges with agent context

---

## 2. LLM CALL SITE

### 2.1 LLM Configuration
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 902-929

**Method:** `def _configure_llm()` (Static Configuration)
**Key Details:**
- Resolves model name (parameter > user default > SDK default)
- Resolves base URL via `resolve_provider_llm_base_url()`
- Creates `LLM` instance with:
  - `model`: Resolved model name
  - `base_url`: Provider-specific or user-configured endpoint
  - `api_key`: From user settings (supports both direct and secret lookups)
  - `usage_id`: Set to 'agent' for tracking

**Return Type:** `LLM` (openhands.sdk.llm.LLM)

### 2.2 LLM + MCP Combined Configuration
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 998-1027

**Method:** `async def _configure_llm_and_mcp()`
**Purpose:** Fully configures both LLM and MCP servers before agent creation

**Execution:**
1. **Lines 1011-1012:** Call `_configure_llm()` to get LLM instance
2. **Lines 1015-1018:** Add system MCP servers (default OpenHands server with Tavily proxy)
3. **Lines 1020-1021:** Merge custom MCP servers from user settings
4. **Lines 1024-1025:** Wrap in SDK's `mcpServers` structure
5. **Return:** Tuple of (configured LLM, MCP config dict)

**MCP Server Format:**
```python
{
  'mcpServers': {
    'server_name': {
      'url': 'server_url',
      'headers': {'X-Session-API-Key': 'key'}
    }
  }
}
```

### 2.3 LLM Metadata for Tracing
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1029-1085

**Method:** `@staticmethod def _apply_server_agent_overrides()`
**Key Details (Lines 1052-1061):**
- For OpenHands managed models (`openhands/*`):
  - Calls `get_llm_metadata()` from `openhands.app_server.utils.llm_metadata`
  - Sets `litellm_extra_body` with metadata for SaaS analytics
  - Metadata includes:
    - `model_name`: The LLM model
    - `llm_type`: Agent vs. condenser vs. other
    - `conversation_id`: For linking requests to conversations
    - `user_id`: For user analytics

**Also applies to condenser LLM (Lines 1063-1083):**
- Updates condenser's `usage_id` to 'condenser'
- Applies same metadata tagging

**Actual LLM Inference:**
- **Where it happens:** In the agent-server, NOT in app-server
- **Invocation:** Agent server receives `StartConversationRequest` with configured LLM
- **API Endpoint:** `POST {agent_server_url}/api/conversations`
- **Agent server responsibility:** Executes the agent loop, making LLM calls internally

---

## 3. TOOL REGISTRY

### 3.1 Tool Loading for Default Agent
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1322-1329

**Tool Registration (DEFAULT agent):**
```python
register_builtins_agents(enable_browser=True)  # Line 1323
tools = get_default_tools(
    enable_browser=True,
    enable_sub_agents=user.agent_settings.enable_sub_agents,  # Line 1326
)
if user.agent_settings.enable_sub_agents:  # Line 1328
    agent_definitions = list(get_registered_agent_definitions())  # Line 1329
```

**Source Module:** `openhands.tools.preset.default`
- **Function:** `get_default_tools(enable_browser=True, enable_sub_agents=False)`
- **Function:** `register_builtins_agents(enable_browser=True)`

**Tool List includes:**
- Shell execution tools
- File operation tools
- Web browsing (if `enable_browser=True`)
- Sub-agent definitions (if `enable_sub_agents=True`)

### 3.2 Tool Loading for Planning Agent
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1317-1321

**Tool Registration (PLAN agent):**
```python
if agent_type == AgentType.PLAN:
    plan_path = None
    if project_dir:
        plan_path = self._compute_plan_path(project_dir, git_provider)
    tools = get_planning_tools(plan_path=plan_path)  # Line 1321
```

**Source Module:** `openhands.tools.preset.planning`
- **Function:** `get_planning_tools(plan_path=None)`
- **Purpose:** Returns tools specific to planning (PLAN.md creation and management)

### 3.3 Tools in Agent Configuration
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1331-1344

**Integration into Agent:**
```python
configured_agent_settings = user.agent_settings.model_copy(
    update={
        'llm': llm,                      # Configured LLM instance
        'tools': tools,                  # Resolved tools list (Line 1337)
        'mcp_config': MCPConfig(...),    # MCP servers
        'agent_context': AgentContext(
            system_message_suffix=effective_suffix,
            secrets=secrets,
        ),
    }
)
agent = configured_agent_settings.create_agent()  # Line 1345
```

**Tool Invocation Flow:**
1. Tools list is passed to `AgentSettings.create_agent()`
2. Agent SDK receives tools and registers them in its internal registry
3. During agent execution (in agent-server), the agent's step function:
   - Calls LLM with tool definitions
   - LLM selects a tool to invoke
   - Agent executes the selected tool
   - Tool output is fed back to LLM
   - Loop continues until task is complete

### 3.4 Skills as Extended Tools
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1402-1411

**Skills Loading:**
```python
if remote_workspace:
    request = await self._load_skills_onto_request(
        request,
        sandbox,
        remote_workspace,
        selected_repository,
        project_dir,
        user.disabled_skills,
    )
```

**Skills Source:**
- **Method:** `_load_skills_onto_request()` (Lines 1415-1442)
- **Calls:** `_load_skills_and_update_agent()` in parent class
- **Calls:** `load_and_merge_all_skills()` (base class, lines 95-157)

**Skills Merge Order:**
1. Public skills (from nue/skills repo)
2. User skills (from ~/.nue/microagents/)
3. Organization skills (from org/.openhands repo)
4. Project/repo skills (from repo .openhands/microagents/)
5. Sandbox skills (from exposed URLs)

**Integration:**
- Skills are merged into `agent.agent_context.skills`
- Skills act like custom tools available to the agent
- Agent can invoke skills during execution

---

## 4. EVENT STREAM PUBLICATION

### 4.1 High-Level Event Architecture
**Event Service Interface:**
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/event/event_service.py`
**Lines:** 16-56

**Abstract Methods:**
- `get_event(conversation_id, event_id)`: Retrieve single event
- `search_events(conversation_id, ...)`: Search events with filters
- `count_events(conversation_id, ...)`: Count matching events
- `save_event(conversation_id, event)`: Store event (internal)
- `batch_get_events(conversation_id, event_ids)`: Get multiple events

**Implementation Classes:**
- `FilesystemEventService`: Stores events in filesystem
- `AwsEventService`: Stores in AWS S3
- `GoogleCloudEventService`: Stores in Google Cloud Storage

### 4.2 Event Callback System
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 384-400

**Event Callback Setup:**
```python
# Setup default processors (Line 385)
processors = request.processors or []

# Ensure SetTitleCallbackProcessor is included (Lines 387-393)
has_set_title_processor = any(
    isinstance(processor, SetTitleCallbackProcessor)
    for processor in processors
)
if not has_set_title_processor:
    processors.append(SetTitleCallbackProcessor())

# Save processors (Lines 396-399)
for processor in processors:
    await self.event_callback_service.save_event_callback(
        EventCallback(
            conversation_id=info.id,
            processor=processor,
        )
    )
```

**Built-in Processors:**
- `SetTitleCallbackProcessor`: Automatically sets conversation title

**Event Callback Service:**
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/event_callback/event_callback_service.py`
**Purpose:** Coordinates event processing and webhook notifications

### 4.3 Message Sending to Conversations
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/app_conversation_router.py`
**Lines:** 425-586

**Endpoint:** `POST /app-conversations/{conversation_id}/send-message`

**Function:** `async def send_message_to_conversation()`

**Key Details:**
1. **Lines 488-502:** Get conversation and sandbox info
2. **Lines 504-525:** Validate sandbox is in RUNNING state
3. **Lines 527-546:** Get agent-server URL from sandbox
4. **Lines 549-564:** Forward message to agent-server:
   ```python
   response = await httpx_client.post(
       f'{agent_server_url}/api/conversations/{conversation_id}/events',
       json={
           'role': request.role,
           'content': content_json,
           'run': request.run,
       },
       headers={'X-Session-API-Key': sandbox.session_api_key},
       timeout=30.0,
   )
   ```
5. **Lines 565-580:** Handle HTTP errors and return response
6. **Lines 582-586:** Return `AppSendMessageResponse` with status

**Message Format:**
- Input: `AppSendMessageRequest`
  - `role`: 'user' or 'assistant'
  - `content`: List of content objects (text, images, etc.)
  - `run`: Boolean to trigger agent step
- Output: `AppSendMessageResponse`
  - `success`: Boolean
  - `sandbox_status`: Current sandbox state
  - `message`: Optional message

### 4.4 Pending Message Processing
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/live_status_app_conversation_service.py`
**Lines:** 1540-1620

**Method:** `async def _process_pending_messages()`

**Purpose:** Process messages queued before conversation was ready

**Execution:**
1. **Lines 1560-1570:** Update pending messages with real conversation_id
2. **Lines 1579-1581:** Get pending messages from database
3. **Lines 1592-1599:** For each pending message:
   - Serialize content to JSON
   - POST to agent-server `/api/conversations/{id}/events`
   - Handle errors gracefully

**Message Source:**
- Messages stored via `PendingMessageService` before agent-server was ready
- Useful for frontend messaging during startup

### 4.5 Stream Start Endpoint
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/app_conversation/app_conversation_router.py`
**Lines:** 869-881

**Endpoint:** `POST /app-conversations/stream-start`

**Function:** `async def stream_app_conversation_start()`
```python
@router.post('/stream-start')
async def stream_app_conversation_start(
    request: AppConversationStartRequest,
    user_context: UserContext = user_context_dependency,
) -> list[AppConversationStartTask]:
    response = StreamingResponse(
        _stream_app_conversation_start(request, user_context),
        media_type='application/json',
    )
    return response
```

**Purpose:**
- Streams conversation startup progress in real-time
- Returns `StreamingResponse` with JSON-encoded start task updates
- Client receives status updates: WORKING → WAITING_FOR_SANDBOX → PREPARING_REPOSITORY → RUNNING_SETUP_SCRIPT → etc.

### 4.6 Event Storage
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/event/event_service.py`
**Lines:** 47-48

**Abstract Method:** `async def save_event(conversation_id: UUID, event: Event)`

**Implementation Details:**
- Events are persisted by concrete `EventService` implementations
- Events come from agent-server and are stored in filesystem/cloud
- Events include:
  - Conversation state changes
  - User messages
  - Agent thoughts and actions
  - Tool executions
  - Errors and status updates

**Event Query Endpoints:**
**File:** `/home/gustavo-silva/GitHub/nue-agentic-work/openhands/app_server/event/event_router.py`
- `GET /api/v1/conversations/{conversation_id}/events/search` (Line 30)
- `GET /api/v1/conversations/{conversation_id}/events/count` (Line 71)
- `GET /api/v1/conversations/{conversation_id}/events` (Line 97)

---

## 5. DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CLIENT (Frontend / API)                                                  │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  POST /app-conversations   │
         │  (start_app_conversation)  │
         └────────────┬───────────────┘
                      │
         ┌────────────▼───────────────┐
         │ AppConversationService     │
         │ (live_status_impl)         │
         │                            │
         │ _start_app_conversation()  │ ──┐
         │ ├─ Wait sandbox start      │   │
         │ ├─ Run setup scripts       │   │
         │ └─ Build conv request      │   │
         └────────────┬───────────────┘   │
                      │                   │
      ┌───────────────▼────────────────┐  │
      │ _build_start_conversation()    │  │
      │                                │  │
      │ ├─ Configure LLM              │  │
      │ ├─ Configure MCP              │  │
      │ ├─ Load & register tools      │  │
      │ │  ├─ get_default_tools()     │  │
      │ │  ├─ get_planning_tools()    │  │
      │ │  └─ get_registered_agents() │  │
      │ ├─ Load skills                │  │
      │ ├─ Load hooks                 │  │
      │ └─ Build Agent                │  │
      │                                │  │
      └───────────────┬────────────────┘  │
                      │                   │
      ┌───────────────▼────────────────┐  │
      │ StartConversationRequest       │  │
      │ (SDK request object)           │  │
      │                                │  │
      │ - agent (configured)           │  │
      │ - workspace                    │  │
      │ - conversation_id              │  │
      │ - initial_message              │  │
      │ - plugins                      │  │
      │ - hooks                        │  │
      └───────────────┬────────────────┘  │
                      │                   │
                      ▼                   │
         ┌────────────────────────────┐   │
         │ httpx.post()               │   │
         │ /api/conversations         │   │
         │ (to agent-server)          │   │
         └────────────┬───────────────┘   │
                      │                   │
                      ▼                   │
         ┌────────────────────────────┐   │
         │ AGENT SERVER               │   │
         │                            │   │
         │ ├─ Create Agent            │   │
         │ ├─ Register Tools          │   │
         │ ├─ Init Agent Loop         │   │
         │ │                          │   │
         │ │ While not done:          │   │
         │ │  ├─ Call LLM             │   │
         │ │  ├─ Parse tool calls     │   │
         │ │  ├─ Execute tools        │   │
         │ │  └─ Emit events          │   │
         │ │                          │   │
         │ └─ Emit events to client   │   │
         │    (WebSocket/SSE)         │   │
         └────────────┬───────────────┘   │
                      │                   │
                      ▼                   │
         ┌────────────────────────────┐   │
         │ EVENT PUBLICATION          │   │
         │                            │   │
         │ ├─ EventCallbackService    │   │
         │ │  └─ Process callbacks    │   │
         │ │     (webhooks, etc)      │   │
         │ │                          │   │
         │ ├─ EventService            │   │
         │ │  └─ Save events          │   │
         │ │     (filesystem/cloud)   │   │
         │ │                          │   │
         │ └─ WebSocket/SSE           │   │
         │    └─ Stream to client     │   │
         └───────────────┬────────────┘   │
                         │                │
                         ▼ (completion)   │
         ┌────────────────────────────┐   │
         │ Return response            │   │
         │ _consume_remaining()◄──────┘   │
         │ background processing     │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │ CLIENT receives response   │
         └────────────────────────────┘

```

---

## 6. KEY FILES SUMMARY

| File | Purpose | Key Functions |
|------|---------|---|
| `app_conversation_router.py` | HTTP REST endpoints | `start_app_conversation()`, `send_message_to_conversation()`, `stream_app_conversation_start()` |
| `live_status_app_conversation_service.py` | Main orchestration | `_start_app_conversation()`, `_build_start_conversation_request_for_user()`, `_configure_llm_and_mcp()` |
| `app_conversation_service_base.py` | Base service | `run_setup_scripts()`, `load_and_merge_all_skills()` |
| `event_service.py` | Event interface | `save_event()`, `search_events()` |
| `event_callback_service.py` | Event processing | Event callback registration and execution |

---

## 7. KEY IMPORTS

- **LLM:** `from openhands.sdk.llm import LLM`
- **Tools:** `from openhands.tools.preset.default import get_default_tools, register_builtins_agents`
- **Tools (Planning):** `from openhands.tools.preset.planning import get_planning_tools, format_plan_structure`
- **Agent SDK:** `from openhands.sdk import Agent, AgentContext`
- **Events:** `from openhands.sdk import Event`
- **Workspace:** `from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace`
- **Skills:** `from openhands.app_server.app_conversation.skill_loader import load_skills_from_agent_server`

---

## 8. Configuration Parameters

### LLM Configuration
- `model`: String identifier (e.g., "gpt-4", "claude-3-opus")
- `base_url`: API endpoint URL
- `api_key`: Authentication token (supports static or lookup)
- `usage_id`: Tracking identifier (default="agent")

### Tool Configuration
- `enable_browser`: Boolean to include web browsing tools
- `enable_sub_agents`: Boolean to enable sub-agent definitions

### MCP Configuration
```python
{
  'mcpServers': {
    'server_name': {
      'url': 'http://...',
      'headers': {...}
    }
  }
}
```

### Agent Types
- `AgentType.DEFAULT`: Full-featured code agent
- `AgentType.PLAN`: Planning-only agent (PLAN.md creation)

---

## 9. Event Types in System

**From Agent Server:**
- Conversation events (state changes)
- User input messages
- Agent output messages
- Thought/reasoning messages
- Tool invocation events
- Tool result events
- Error events

**Stored by EventService:**
- Filesystem: `.openhands/conversations/{conv_id}/events/`
- AWS S3: `s3://bucket/conversations/{conv_id}/events/`
- Google Cloud: `gs://bucket/conversations/{conv_id}/events/`

**Processed by EventCallbackService:**
- Webhooks (external integrations)
- Title auto-setting
- Custom processors

---

## 10. Connection Points for Developers

### To Add a New Tool
1. Create tool function or class in `openhands/tools/`
2. Register in `get_default_tools()` or `get_planning_tools()`
3. Tool becomes available to all agents automatically

### To Add Custom LLM Logic
1. Implement at line 902-929 in `live_status_app_conversation_service.py`
2. Override `_configure_llm()` or `_configure_llm_and_mcp()`
3. Return configured `LLM` instance

### To Add Event Processing
1. Create processor inheriting from callback base class
2. Register in event callback setup (line 384-400)
3. Processor receives events and can take action

### To Monitor Agent Execution
1. Hook into event stream via WebSocket/SSE
2. Query `/api/v1/conversations/{id}/events/search`
3. Filter by event kind or timestamp
