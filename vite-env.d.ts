/// <reference types="vite/client" />

/**
 * Absolute filesystem path to the bundled extensions skills directory,
 * injected by Vite at build time via `define` in vite.config.ts.
 * Empty string in library builds; always a real path in app/dev builds.
 */
declare const __EXTENSIONS_SKILLS_DIR__: string;
