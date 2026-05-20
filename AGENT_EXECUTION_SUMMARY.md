# Agent Execution Lifecycle - Quick Reference

## Quick Navigation

### 1. Agent Step Function (Main Execution Loop)

| Entry Point | File & Line | Purpose |
|-------------|-------------|---------|
| **REST API Entry** | `app_conversation_router.py:360` | `start_app_conversation()` - HTTP endpoint |
| **Service Orchestration** | `live_status_app_conversation_service.py:232` | `start_app_conversation()` - async generator wrapper |
| **Main Implementation** | `live_status_app_conversation_service.py:241` | `_start_app_conversation()` - full setup flow |
| **Agent Config Builder** | `live_status_app_conversation_service.py:1214` | `_build_start_conversation_request_for_user()` - agent setup |

**Execution Sequence:**
```
POST /app-conversations
  → start_app_conversation()
  → _start_app_conversation()
  → _wait_for_sandbox_start()
  → run_setup_scripts()
  → _build_start_conversation_request_for_user()
  → httpx POST /api/conversations (to agent-server)
  → save_app_conversation_info()
  → _process_pending_messages()
```

---

### 2. LLM Call Sites

| Location | File & Line | Function |
|----------|-------------|----------|
| **LLM Configuration** | `live_status_app_conversation_service.py:902` | `_configure_llm()` - creates LLM instance |
| **LLM + MCP Setup** | `live_status_app_conversation_service.py:998` | `_configure_llm_and_mcp()` - full LLM+MCP config |
| **LLM Metadata** | `live_status_app_conversation_service.py:1052` | LLM tracing metadata for analytics |
| **Actual Inference** | Agent Server (NOT app-server) | Agent executes LLM calls in agent-server loop |

**Key Configuration:**
- **Model:** Resolved from parameter > user default > SDK default
- **Base URL:** Provider-specific or user-configured
- **API Key:** From user settings (static or lookup)
- **Usage ID:** "agent" for tracking
- **Metadata:** For SaaS analytics on openhands/* models

---

### 3. Tool Registry

| Agent Type | File & Line | Function |
|------------|-------------|----------|
| **DEFAULT Agent** | `live_status_app_conversation_service.py:1322` | `get_default_tools(enable_browser=True, ...)` |
| **PLAN Agent** | `live_status_app_conversation_service.py:1321` | `get_planning_tools(plan_path=...)` |
| **Sub-agents** | `live_status_app_conversation_service.py:1328` | `get_registered_agent_definitions()` |
| **Agent Creation** | `live_status_app_conversation_service.py:1331` | Tools integrated into `AgentSettings` |
| **Skills Loading** | `live_status_app_conversation_service.py:1402` | `_load_skills_onto_request()` - custom skills |

**Tool Sources:**
1. Core tools (from `openhands.tools.preset`)
2. Skills from multiple sources:
   - Public skills (GitHub)
   - User skills (~/.nue/microagents/)
   - Org skills (.openhands repo)
   - Project skills (repo .openhands/microagents/)
   - Sandbox skills (exposed URLs)

---

### 4. Event Stream Publication

| Component | File & Line | Purpose |
|-----------|-------------|---------|
| **Event Service Interface** | `event_service.py:16` | Abstract event operations |
| **Event Storage** | `event_service.py:47` | `save_event()` - persist events |
| **Event Querying** | `event_router.py:30,71,97` | Search, count, batch get events |
| **Event Callbacks** | `live_status_app_conversation_service.py:384` | `SetTitleCallbackProcessor` and custom processors |
| **Message Sending** | `app_conversation_router.py:436` | `send_message_to_conversation()` - forward to agent-server |
| **Stream Endpoint** | `app_conversation_router.py:869` | `stream_app_conversation_start()` - real-time progress |
| **Pending Messages** | `live_status_app_conversation_service.py:1540` | `_process_pending_messages()` - queued messages |

**Event Flow:**
```
Agent Server (running in sandbox)
  → Emits events
  → EventCallbackService (processes callbacks)
  → EventService (stores events)
  → EventRouter (REST endpoints for querying)
  → Client (receives via WebSocket/SSE)
```

---

## File Summary Table

| File | Key Functions | Lines |
|------|---|---|
| **app_conversation_router.py** | `start_app_conversation()`, `send_message_to_conversation()`, `stream_app_conversation_start()` | 360, 436, 869 |
| **live_status_app_conversation_service.py** | `_start_app_conversation()`, `_build_start_conversation_request_for_user()`, `_configure_llm_and_mcp()`, `_load_skills_onto_request()` | 232, 1214, 998, 1415 |
| **app_conversation_service_base.py** | `run_setup_scripts()`, `load_and_merge_all_skills()` | 245, 95 |
| **event_service.py** | Abstract service interface | 16 |
| **event_callback_service.py** | Event processor registration | - |
| **event_router.py** | REST endpoints for event querying | 30, 71, 97 |

---

## Critical Code Paths

### Path 1: Start Conversation
```
POST /app-conversations (line 360)
  ├─ _start_app_conversation() (line 241)
  ├─ _wait_for_sandbox_start() (line 269)
  ├─ run_setup_scripts() (line 294, base class line 245)
  ├─ _build_start_conversation_request_for_user() (line 305)
  │  ├─ _configure_llm_and_mcp() (line 1292)
  │  ├─ get_default_tools() or get_planning_tools() (line 1321 or 1324)
  │  ├─ _load_skills_onto_request() (line 1403)
  │  └─ create_agent() (line 1345)
  ├─ httpx POST /api/conversations (line 337)
  └─ _process_pending_messages() (line 1540)
```

### Path 2: Send Message
```
POST /app-conversations/{id}/send-message (line 436)
  ├─ Validate sandbox is RUNNING (line 504)
  ├─ Get agent-server URL (line 534)
  └─ httpx POST /api/conversations/{id}/events (line 551)
```

### Path 3: Stream Start Progress
```
POST /app-conversations/stream-start (line 869)
  └─ StreamingResponse(_stream_app_conversation_start()) (line 877)
     └─ Yields AppConversationStartTask with status updates
```

---

## Key Data Structures

### StartConversationRequest (to agent-server)
```python
{
    'agent': Agent,                    # Configured agent
    'workspace': LocalWorkspace,        # Working directory
    'conversation_id': UUID,            # Unique ID
    'initial_message': SendMessageRequest,  # First message
    'plugins': List[PluginSource],      # Plugins
    'hook_config': HookConfig,          # Git hooks
    'agent_definitions': List,          # Sub-agents
}
```

### LLM Instance
```python
{
    'model': str,           # Model name (e.g., "gpt-4")
    'base_url': str,        # API endpoint
    'api_key': SecretStr,   # Authentication
    'usage_id': str,        # Tracking identifier
}
```

### MCP Configuration
```python
{
    'mcpServers': {
        'server_name': {
            'url': str,
            'headers': Dict[str, str],
        }
    }
}
```

---

## Important Constants & Enums

### Agent Types
- `AgentType.DEFAULT` - Full-featured code agent
- `AgentType.PLAN` - Planning-only agent (creates PLAN.md)

### Sandbox Status
- `RUNNING` - Ready for messages
- `PAUSED` - Closed but can resume
- `STARTING` - Initializing
- `ERROR` - In error state
- `MISSING` - Archive/deleted

### Event Processing
- `EventService` implementations: Filesystem, AWS S3, Google Cloud
- `EventCallback` processors: SetTitleCallbackProcessor, custom

---

## Developer Integration Points

### 1. Add New Tool
**Where:** `openhands/tools/preset/default.py` or `planning.py`
**How:** Register in `get_default_tools()` or `get_planning_tools()`

### 2. Customize LLM
**Where:** `live_status_app_conversation_service.py:902`
**How:** Override `_configure_llm()`

### 3. Add Event Processing
**Where:** Event callback system (line 384)
**How:** Create processor class, register in setup

### 4. Monitor Execution
**How:** Query `/api/v1/conversations/{id}/events/search` or WebSocket

---

## Performance Considerations

1. **LLM Configuration:** Minimal overhead (1 instance per conversation)
2. **Tool Registration:** Happens at agent creation time (cached thereafter)
3. **Skill Loading:** Async, with error fallback (continues without skills if error)
4. **Event Processing:** Background task, doesn't block response
5. **Pending Messages:** Processed sequentially to preserve order

---

## Error Handling

| Error Type | Where | Recovery |
|------------|-------|----------|
| Sandbox not ready | `_wait_for_sandbox_start()` | Retry with exponential backoff |
| LLM config failure | `_configure_llm()` | Use SDK defaults |
| Skill load failure | `load_and_merge_all_skills()` | Continue without skills |
| Hook load failure | `_load_hooks_from_workspace()` | Continue without hooks |
| Agent-server unreachable | `httpx POST /api/conversations` | Return 502 error |
| Message send failure | `send_message_to_conversation()` | Return HTTP error |
