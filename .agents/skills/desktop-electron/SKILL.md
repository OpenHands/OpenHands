---
name: desktop-electron
description: This skill should be used when the user asks to "change the desktop app", "package Electron", "build a universal macOS app", "bundle Node or uv", "fix Electron startup", or changes electron/, electron-builder.config.mjs, desktop workflows, bundled runtimes, or desktop process startup.
---

# Electron Desktop

Preserve desktop startup reliability and packaging invariants across development, installed single-architecture builds, and universal macOS builds.

## Workflow

1. Read `references/guide.md` before changing Electron startup, branding, packaging hooks, runtime resources, or desktop CI.
2. Test cold-start readiness independently from ingress readiness.
3. Keep packaged dependency stripping and the explicit runtime package closure synchronized.
4. Validate bundled Node and uv paths on every affected platform and architecture.
5. Test installed artifacts outside the repository tree so local `node_modules` cannot hide missing packaged dependencies.

## Reference

- **`references/guide.md`** — Cold-start coordination, loading feedback, node_modules stripping, universal runtime layout, macOS development branding, and bundled Node/npm/npx behavior.
