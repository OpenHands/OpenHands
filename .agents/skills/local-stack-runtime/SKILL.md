---
name: local-stack-runtime
description: This skill should be used when the user asks to "change the dev stack", "add a runtime service", "change the launcher", "update Docker", "bump Agent Server", "change ingress routing", or changes scripts/dev-*.mjs, runtime-services metadata, config/defaults.json, Docker, automation startup, process shutdown, or local authentication.
---

# Local Stack and Runtime Services

Maintain the shared configuration and launch plumbing across Vite/static development, Agent Server, automation, ingress, the published binary, and the all-in-one Docker image.

## Workflow

1. Read `references/guide.md` before changing launchers, runtime service metadata, version pins, Docker composition, auth generation, or process management.
2. Treat `config/defaults.json` and the injected `<RUNTIME_SERVICES>` block as authoritative.
3. Preserve coordinated process-tree shutdown and fail fast on missing prerequisites.
4. Keep Agent Server and automation versions compatible and authentication keys correctly scoped.
5. Update documentation and version-drift tests when changing pins or launch commands.
6. Validate each affected launch mode rather than assuming Vite, static, npm, Docker, and Electron behave identically.

## Reference

- **`references/guide.md`** — Runtime-services schema and prompt plumbing, launcher environment variables, ingress routes, version synchronization, security, Docker build architecture, README synchronization, and spec-tag conventions.
