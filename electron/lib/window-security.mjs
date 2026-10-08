/**
 * Renderer security options shared by every Electron window.
 *
 * `nodeIntegration: false` and `contextIsolation: true` keep Node and the
 * preload's scope out of page JavaScript, but neither confines the renderer
 * process itself. `sandbox: true` is the OS-level half: the renderer runs in
 * the Chromium sandbox, so code that escapes the JS boundary still faces the
 * kernel restrictions rather than the privileges of the user running the app.
 *
 * Electron enables the sandbox by default for renderers, and has changed that
 * default before. Relying on it makes a security property of these windows
 * depend on which Electron the build resolved, and the value is silent either
 * way: a sandboxed and an unsandboxed window look identical at runtime. Naming
 * it here turns an inherited default into a stated one, and gives the test
 * below something to assert.
 *
 * A sandboxed preload cannot use ESM and cannot require Node built-ins beyond
 * the polyfilled subset. `preload.cjs` is already written to that contract:
 * it is CommonJS and imports only `contextBridge` and `ipcRenderer`, both of
 * which a sandboxed preload keeps.
 */
export const SECURE_WEB_PREFERENCES = Object.freeze({
  nodeIntegration: false,
  contextIsolation: true,
  sandbox: true,
});
