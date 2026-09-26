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

### SC-005: Conversation condensation

- [x] `/condense` shall request condensation through the typed conversation client.
- [x] Backends returning HTTP 404, 405, or 501 shall produce a localized unsupported message; other failures shall produce a localized generic failure.
