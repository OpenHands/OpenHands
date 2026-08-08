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
