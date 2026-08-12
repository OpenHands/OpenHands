# Slash Command Specs

---

### SC-001: One discoverable command registry

- [x] The autocomplete menu shall combine built-in commands with skill-derived triggers.
- [x] Built-in commands shall be available on both local and cloud backends.
- [x] `/new` shall reuse the current runtime and support synchronous local creation as well as asynchronous cloud creation.

### SC-002: Inline help

- [x] `/help` shall render only built-in slash commands in the chat area.
- [x] Built-in commands shall have localized descriptions, and help shall mention the `/` autocomplete menu.
- [x] Built-in help shall not depend on the skill catalog.

### SC-003: Loaded extensions

- [x] `/skills` shall render workspace skills, conversation hooks, and enabled MCP servers in the chat area.
- [x] Disabled MCP servers shall not be presented as loaded.
- [x] The output shall remain anchored to the event that preceded the command.
- [x] A failed hooks or settings refresh shall not hide extension data that is still available from other sources.
