import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Build the in-app path to the `/launch` screen for the given plugins, so the
 * Plugins UI can start a conversation with a plugin by reusing the existing
 * launch flow. Inverse of `parsePluginsFromUrl` in `src/routes/launch.tsx` —
 * keep the base64-encoded JSON `plugins` format (and its UTF-8 handling) in
 * sync with that decoder.
 *
 * The JSON is UTF-8 encoded before base64 so non-Latin-1 parameter values
 * survive; `decodePluginsPayload` performs the inverse.
 *
 * The base64 payload can contain `+`, `/`, and `=`; encoding it through
 * `URLSearchParams` percent-escapes those so `/launch` reads back the exact
 * string (a raw `+` would otherwise be decoded as a space and break `atob`).
 */
export function encodePluginsPayload(plugins: PluginSpec[]): string {
  const bytes = new TextEncoder().encode(JSON.stringify(plugins));
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

/**
 * Decode a base64 `plugins` payload back into a JSON string, treating the
 * underlying bytes as UTF-8. ASCII-only payloads from before UTF-8 support
 * remain compatible because ASCII is a subset of UTF-8.
 */
export function decodePluginsPayload(encoded: string): string {
  const binary = atob(encoded);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function buildPluginLaunchPath(plugins: PluginSpec[]): string {
  const params = new URLSearchParams({
    plugins: encodePluginsPayload(plugins),
  });
  return `/launch?${params.toString()}`;
}
