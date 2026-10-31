/// <reference types="react-scripts" />

// Injected by vite.config.ts `define` — absolute path to the
// @openhands/extensions skills directory in node_modules, or an
// empty string in library builds.
declare const __EXTENSIONS_SKILLS_DIR__: string;

// Injected by vite.config.ts `define` — the operator's feature policy resolved
// from AGENT_CANVAS_FEATURES / AGENT_CANVAS_FEATURES_FILE under the Vite dev
// server; null in builds, where static-server injects it at runtime instead.
declare const __AGENT_CANVAS_DEV_FEATURE_POLICY__: {
  version: 1;
  disabled: string[];
} | null;
