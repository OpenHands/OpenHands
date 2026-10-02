# MCP Settings Specs

---

### MCP-001: Sparse mutations preserve sibling servers

- [x] Adding, updating, or deleting an MCP server shall issue exactly one
      dedicated MCP settings request containing only the affected server.
      On a cloud backend, an add or update first reads the stored catalog and
      writes it back in full, because the cloud replaces the whole catalog for
      an `mcp_config` map without `null` entries (OHE-3248).
- [x] An MCP mutation shall never use redacted or encrypted settings snapshots
      as its mutation base. On a cloud backend, untouched sibling servers are
      resent verbatim from the redacted settings read (the cloud restores their
      secrets by key); redacted values are still never used as input for the
      mutated server.
- [x] Untouched sibling servers and credentials shall survive add, update,
      delete, concurrent different-key updates, and failed mutations.

### MCP-002: Secret patches preserve user intent

- [x] Omitting an unchanged secret shall preserve its stored value.
- [x] Supplying a secret shall replace its stored value.
- [x] Explicitly clearing a supported secret field shall send `null`.
- [x] The display-only redaction sentinel shall never be sent as mutation data.

### MCP-003: Settings map keys are stable MCP identities

- [x] Rendering order and transport grouping shall not change persistence
      identity.
- [x] Deletion shall address the settings map key without matching URL, command,
      or arguments.
- [x] Rename shall use one atomic map patch, reject collisions, and reject
      hidden-secret renames that could lose credentials.
