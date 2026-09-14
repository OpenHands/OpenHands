# Demo page Canvas Extension

This fixture is a standalone, dependency-free implementation of manifest
schema 1 and host API 1. Its JavaScript file is already a self-contained browser
ES module, so an Agent Server test implementation can install this directory and
serve `extension.js` directly from the authenticated bundle endpoint.

For the browser-only installation flow that works before those Agent Server
endpoints exist, see
[`docs/CANVAS_EXTENSIONS_TESTING.md`](../../../../docs/CANVAS_EXTENSIONS_TESTING.md).
