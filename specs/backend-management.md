# Backend Management Specs

---

### BM-001: Auto-switch on connect
- [x] Adding a backend shall automatically switch the active selection to it.

### BM-002: Switching backends keeps the user on the same page
- [x] Switching backends shall redirect to the same section but on the new backend. The user shall never see stale data from the previous backend.

### BM-003: Fallback on active backend removal
- [x] Removing the currently active backend shall fall back to a remaining local backend. The user shall never be left without an active backend.

### BM-004: Display the active backend's execution mode
- [x] The Manage backends surface shall show each local backend's execution boundary (local or Docker) next to its version badge.
- [x] The mode shall be read from the agent server's reported runtime field (`conversation_runtime`, or `execution_runtime` when present); no new endpoint or contract is added.
- [x] A single resolver (`getBackendExecutionMode`) shall be the only reader of that field, so the displayed mode and the workspace the server is asked for during conversation creation can never disagree.
- [x] When the server omits the field, no mode badge shall render and the row is unchanged.
- [x] Cloud backends expose no `/server_info`, so they render no mode badge even when they share a host and key with a local backend.
- [x] The badge shall refresh when the backend health poll observes new server info, so a server that restarts in a different mode is reflected without reopening the modal.
- [x] The mode label shall be localized through `src/i18n/translation.json`.