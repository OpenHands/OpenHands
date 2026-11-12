import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Encode text as UTF-8 before base64. Plain `btoa` operates on one-byte
 * binary strings and throws `InvalidCharacterError` for non-Latin-1 text,
 * which `PluginSpec.parameters` can contain (#16735). Chunks the byte string
 * so large payloads do not blow the argument limit of `String.fromCharCode`.
 */
export function encodePluginsBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/**
 * Inverse of `encodePluginsBase64`: decode the base64 bytes as UTF-8 instead
 * of reading them as Latin-1, which silently corrupted Unicode parameter
 * values (#16735). Legacy ASCII payloads decode unchanged.
 */
export function decodePluginsBase64(base64: string): unknown {
  const binary = atob(base64);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes));
}

/**
 * Build the in-app path to the `/launch` screen for the given plugins, so the
 * Plugins UI can start a conversation with a plugin by reusing the existing
 * launch flow. Inverse of `parsePluginsFromUrl` in `src/routes/launch.tsx` —
 * keep the base64-encoded JSON `plugins` format in sync with that decoder,
 * which uses `decodePluginsBase64` from this module.
 *
 * The base64 payload can contain `+`, `/`, and `=`; encoding it through
 * `URLSearchParams` percent-escapes those so `/launch` reads back the exact
 * string (a raw `+` would otherwise be decoded as a space and break `atob`).
 */
export function buildPluginLaunchPath(plugins: PluginSpec[]): string {
  const encoded = encodePluginsBase64(JSON.stringify(plugins));
  const params = new URLSearchParams({ plugins: encoded });
  return `/launch?${params.toString()}`;
}
