import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Build the in-app path to the `/launch` screen for the given plugins, so the
 * Plugins UI can start a conversation with a plugin by reusing the existing
 * launch flow. Inverse of `parsePluginsFromUrl` in `src/routes/launch.tsx` —
 * keep the base64-encoded JSON `plugins` format in sync with that decoder.
 *
 * The base64 payload can contain `+`, `/`, and `=`; encoding it through
 * `URLSearchParams` percent-escapes those so `/launch` reads back the exact
 * string (a raw `+` would otherwise be decoded as a space and break `atob`).
 */
export function buildPluginLaunchPath(plugins: PluginSpec[]): string {
  const encoded = btoa(JSON.stringify(plugins));
  const params = new URLSearchParams({ plugins: encoded });
  return `/launch?${params.toString()}`;
}
