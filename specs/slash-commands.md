# Slash Command Specs

---

### SC-001: One discoverable command registry

- [x] The autocomplete menu and `/help` shall use the same built-in and skill-derived command registry.
- [x] Built-in commands shall be available on both local and cloud backends.
- [x] `/new` shall reuse the current runtime and support synchronous local creation as well as asynchronous cloud creation.

### SC-002: Inline help

- [x] `/help` shall render every available built-in and skill-derived slash command in the chat area.
- [x] Built-in commands shall have localized descriptions, and help shall mention the `/` autocomplete menu.
- [x] Built-in help shall remain available when the skill catalog cannot be refreshed.

### SC-003: Loaded extensions

- [x] `/skills` shall render workspace skills, conversation hooks, and enabled MCP servers in the chat area.
- [x] Disabled MCP servers shall not be presented as loaded.
- [x] The output shall remain anchored to the event that preceded the command.
- [x] A failed hooks or settings refresh shall not hide extension data that is still available from other sources.

### SC-004: Feedback

- [x] `/feedback` shall open the anonymous feedback form in a protected new browser tab without sending a user message to the agent.

### SC-005: Conversation condensation

- [x] `/condense` shall request condensation through the typed conversation client.
- [x] Backends returning HTTP 404, 405, or 501 shall produce a localized unsupported message; other failures shall produce a localized generic failure.
