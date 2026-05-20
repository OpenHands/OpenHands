# OpenHands Agent Execution Lifecycle - Documentation Index

## Overview

This documentation package provides comprehensive coverage of the OpenHands agent execution lifecycle, mapping all critical entry points where:
- Agent steps are executed
- LLM models are invoked
- Tools are registered and discovered
- Events are published to clients

## Documentation Files

### 1. AGENT_EXECUTION_LIFECYCLE.md (Comprehensive)
**Size:** Detailed technical reference (4000+ lines of organized content)
**Audience:** Developers implementing new features or debugging
**Contents:**
- Complete mapping of all entry points with file paths and line numbers
- Detailed execution flows with code examples
- Data flow diagrams
- Configuration parameters
- Event types and storage mechanisms
- Connection points for customization
- Key imports and modules

**Best for:** Deep understanding, implementation details, integration work

### 2. AGENT_EXECUTION_SUMMARY.md (Quick Reference)
**Size:** Concise navigation guide (200-300 lines)
**Audience:** Developers needing quick lookups or overview
**Contents:**
- Quick navigation tables
- File summary table
- Critical code paths in tree format
- Key data structures
- Important constants & enums
- Developer integration points
- Error handling matrix

**Best for:** Quick lookups, navigation, decision-making

---

## Finding What You Need

### If you need to understand...

**Agent Step Function / Run Loop:**
- Quick Reference: AGENT_EXECUTION_SUMMARY.md → Section 1 & Critical Code Paths
- Detailed: AGENT_EXECUTION_LIFECYCLE.md → Section 1
- Key Files:
  - `openhands/app_server/app_conversation/app_conversation_router.py:360`
  - `openhands/app_server/app_conversation/live_status_app_conversation_service.py:232`

**LLM Call Sites:**
- Quick Reference: AGENT_EXECUTION_SUMMARY.md → Section 2
- Detailed: AGENT_EXECUTION_LIFECYCLE.md → Section 2
- Key Files:
  - `openhands/app_server/app_conversation/live_status_app_conversation_service.py:902` (LLM config)
  - `openhands/app_server/app_conversation/live_status_app_conversation_service.py:998` (LLM + MCP)

**Tool Registry:**
- Quick Reference: AGENT_EXECUTION_SUMMARY.md → Section 3
- Detailed: AGENT_EXECUTION_LIFECYCLE.md → Section 3
- Key Files:
  - `openhands/tools/preset/default.py` (core tools)
  - `openhands/tools/preset/planning.py` (planning tools)
  - `openhands/app_server/app_conversation/live_status_app_conversation_service.py:1315`

**Event Stream Publication:**
- Quick Reference: AGENT_EXECUTION_SUMMARY.md → Section 4
- Detailed: AGENT_EXECUTION_LIFECYCLE.md → Section 4
- Key Files:
  - `openhands/app_server/event/event_service.py` (interface)
  - `openhands/app_server/event/event_router.py` (REST endpoints)
  - `openhands/app_server/app_conversation/app_conversation_router.py:436` (send message)

---

## Architecture Overview

### Three-Layer Architecture

```
┌────────────────────────────────────────┐
│ Layer 1: Frontend / Client             │
│ (Web UI, API Consumer)                 │
└─────────────┬──────────────────────────┘
              │ HTTP REST / WebSocket
              ▼
┌────────────────────────────────────────┐
│ Layer 2: App Server                    │
│ (openhands/app_server/)                │
│                                        │
│ ├─ Conversation Router (REST endpoints)│
│ ├─ Conversation Service (orchestration)│
│ ├─ Event Service (persistence)         │
│ └─ Callback Service (processing)       │
└─────────────┬──────────────────────────┘
              │ HTTP REST (SDK request)
              ▼
┌────────────────────────────────────────┐
│ Layer 3: Agent Server (in Sandbox)     │
│ (Separate process)                     │
│                                        │
│ ├─ Agent Instance                      │
│ ├─ Tool Registry                       │
│ ├─ Agent Loop (step function)          │
│ ├─ LLM Calls (to provider)             │
│ └─ Event Emission (back to app-server) │
└────────────────────────────────────────┘
```

### Data Flow Summary

1. **Initialization:**
   - Client calls `/api/v1/app-conversations` (start conversation)
   - App-server configures LLM, tools, secrets, workspace
   - Sends `StartConversationRequest` to agent-server

2. **Execution:**
   - Agent-server creates agent instance with tools
   - Enters agent loop (step function)
   - For each iteration:
     - Calls LLM with tool definitions
     - LLM selects tool or generates response
     - Executes tool in workspace
     - Collects output
     - Emits event back to app-server
   - Continues until goal achieved

3. **Event Publishing:**
   - App-server receives events from agent-server
   - EventCallbackService processes callbacks (webhooks, title setting, etc.)
   - EventService persists to storage (filesystem/S3/GCS)
   - Client receives via WebSocket or REST query

---

## Key Concepts

### StartConversationRequest
The central data structure passed from app-server to agent-server. Contains:
- **Agent:** Pre-configured agent with tools and LLM
- **Workspace:** Working directory path
- **Secrets:** API keys and credentials
- **MCP Config:** Model Context Protocol servers
- **Plugins:** User-selected plugins
- **Hooks:** Git hooks configuration
- **Initial Message:** First message to send to agent

### Agent Configuration Pipeline
```
User Settings
  ↓
Resolve LLM (model, base_url, api_key)
  ↓
Configure MCP (servers)
  ↓
Load Tools (default or planning)
  ↓
Load Skills (project-specific)
  ↓
Load Hooks (git integration)
  ↓
Create Agent Instance
  ↓
Apply Server Overrides (metadata, tracing)
  ↓
Send to Agent Server
```

### Tool Invocation Flow (in Agent Server)
```
While not done:
  1. Get current conversation state
  2. Call LLM with:
     - System prompt
     - Messages history
     - Tool definitions
  3. LLM returns:
     - Message content OR
     - Tool invocation request
  4. If tool invocation:
     - Execute tool
     - Collect output
     - Add to messages history
  5. Emit event to app-server
  6. Loop
```

---

## Common Tasks

### Add a New Tool
1. Create tool function in `openhands/tools/preset/`
2. Register in `get_default_tools()` or `get_planning_tools()`
3. Tool automatically available to all agents

**Reference:** AGENT_EXECUTION_SUMMARY.md → Developer Integration Points

### Customize LLM Configuration
1. Modify `_configure_llm()` in `live_status_app_conversation_service.py:902`
2. Override base URL resolution, model selection, or metadata
3. Return configured `LLM` instance

**Reference:** AGENT_EXECUTION_LIFECYCLE.md → Section 2.1-2.3

### Add Event Processing
1. Create processor class (inherit from base)
2. Register in event callback setup (line 384)
3. Processor receives and acts on events (webhooks, etc)

**Reference:** AGENT_EXECUTION_LIFECYCLE.md → Section 4.2

### Monitor Agent Execution
1. Hook into WebSocket stream (real-time)
2. Or query `/api/v1/conversations/{id}/events/search` (historical)
3. Filter by event kind, timestamp, etc

**Reference:** AGENT_EXECUTION_LIFECYCLE.md → Section 4.5-4.6

---

## Key Files Quick Reference

| Component | File | Purpose |
|-----------|------|---------|
| REST API | `app_conversation_router.py` | HTTP endpoints |
| Orchestration | `live_status_app_conversation_service.py` | Main logic |
| Base Service | `app_conversation_service_base.py` | Shared functionality |
| Event Interface | `event_service.py` | Abstract service |
| Event Endpoints | `event_router.py` | REST queries |
| Event Callbacks | `event_callback_service.py` | Processing |
| Default Tools | `openhands/tools/preset/default.py` | Tool definitions |
| Planning Tools | `openhands/tools/preset/planning.py` | Planning tools |

---

## Common Debugging

### Agent not responding to messages
1. Check: POST `/app-conversations/{id}/send-message` returns error
2. Verify: Sandbox status is RUNNING
3. Check: Agent-server `/api/conversations/{id}` is reachable

**Details:** AGENT_EXECUTION_LIFECYCLE.md → Section 4.3

### Tools not available
1. Check: Tool registered in `get_default_tools()` or `get_planning_tools()`
2. Verify: AgentSettings includes tools in `agent.tools`
3. Check: Agent-server loaded tools correctly

**Details:** AGENT_EXECUTION_LIFECYCLE.md → Section 3

### LLM not configured
1. Check: User has LLM settings configured
2. Verify: `_configure_llm()` resolves model and base_url
3. Check: API key available and not expired

**Details:** AGENT_EXECUTION_LIFECYCLE.md → Section 2

### Events not saving
1. Check: EventService implementation (Filesystem/S3/GCS)
2. Verify: Storage backend is accessible
3. Check: Conversation ID properly passed to `save_event()`

**Details:** AGENT_EXECUTION_LIFECYCLE.md → Section 4.6

---

## Integration with Enterprise

If using the enterprise module:
- Agent execution lifecycle is identical
- Enterprise adds auth, billing, analytics layers
- Secrets may come from Keycloak instead of database
- Events may trigger enterprise integrations (Jira, Slack, etc)

---

## Performance Notes

1. **Conversation Startup:** ~5-30 seconds
   - Sandbox startup: 5-15s
   - Setup scripts: 2-10s
   - Skill loading: 1-5s

2. **Message Latency:** <100ms (app-server only)
   - Actual LLM latency: seconds (in agent-server)

3. **Event Processing:** Background task
   - Does not block response
   - Processed asynchronously

4. **Skill Loading:** Cached after first load
   - Subsequent conversations faster

---

## Testing Entry Points

### Unit Tests
- `tests/unit/test_app_conversation_*.py` - Service tests
- `tests/unit/test_event_*.py` - Event tests

### Integration Tests
- End-to-end conversation flow
- Sandbox startup and teardown
- Event emission and storage

### Debugging
- Enable debug logging: `logger.setLevel(logging.DEBUG)`
- Check `.openhands/logs/` for detailed traces
- Monitor `/tmp/openhands-log.txt` for startup errors

---

## Related Documentation

- **OpenHands SDK:** Tool definition, Agent, Workspace classes
- **Agent Server:** Conversation implementation, step function
- **Sandbox Management:** Sandbox startup, image building
- **Event Callback:** Webhook integration, custom processors
- **User Settings:** LLM profiles, agent preferences

---

## Document Maintenance

Last Updated: [Current Date]
Coverage: OpenHands V1 App Server
Scope: Agent execution lifecycle only

For updates:
1. Update AGENT_EXECUTION_LIFECYCLE.md for detailed changes
2. Update AGENT_EXECUTION_SUMMARY.md for navigation/quick-ref changes
3. This index as needed
