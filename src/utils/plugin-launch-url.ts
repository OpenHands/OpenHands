import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

export function encodePluginLaunchPayload(plugins: PluginSpec[]): string {
  return bytesToBase64(new TextEncoder().encode(JSON.stringify(plugins)));
}

export function decodePluginLaunchPayload(payload: string): unknown {
  const binary = atob(payload);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));

  let json: string;
  try {
    json = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    // Preserve links generated before plugin payloads were encoded as UTF-8.
    json = binary;
  }

  return JSON.parse(json);
}

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
  const encoded = encodePluginLaunchPayload(plugins);
  const params = new URLSearchParams({ plugins: encoded });
  return `/launch?${params.toString()}`;
}
