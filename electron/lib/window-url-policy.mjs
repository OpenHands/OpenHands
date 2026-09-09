/**
 * URL classification for window.open() and popup navigation in the Electron
 * shell.
 *
 * The renderer renders untrusted links (chat / agent markdown) with
 * `target="_blank"`, which routes through `setWindowOpenHandler` in
 * main.mjs. Classifying those URLs by string prefix
 * (`url.startsWith("http://localhost")`) lets attacker-controlled hosts
 * through as "app URLs" — `http://localhost.evil.com` satisfies the prefix,
 * and so does `http://localhost@evil.com` (userinfo). Either one would open
 * remote content in a chromeless native window that visually belongs to the
 * app. Schemes outside a small allowlist must also never be handed to
 * `shell.openExternal`, which forwards them to OS protocol handlers.
 * Both classifications therefore go through URL parsing, not prefixes.
 */

const LOOPBACK_HOSTNAMES = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

// Schemes `shell.openExternal` may be given. Kept in sync with the `href`
// protocols the renderer's markdown sanitizer accepts
// (`MARKDOWN_SANITIZE_SCHEMA` in
// src/components/features/markdown/markdown-renderer.tsx) — a link the
// renderer is willing to render but this policy denies becomes a click that
// silently does nothing. `mailto:` and `tel:` reach the mail/dialer handler;
// `file:`, `smb:` and custom scheme handlers stay denied.
const EXTERNAL_PROTOCOLS = new Set(["http:", "https:", "mailto:", "tel:"]);

/**
 * True for http(s) URLs whose host is the loopback app server — the only
 * URLs that may open inside an Electron window.
 */
export function isLoopbackAppUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      LOOPBACK_HOSTNAMES.has(url.hostname)
    );
  } catch {
    return false;
  }
}

/**
 * True for URLs that may be handed to the OS via `shell.openExternal` — the
 * allowlisted schemes above. Everything else (file:, smb:, custom scheme
 * handlers, unparsable strings) must be denied, not delegated to the OS.
 */
export function isExternalBrowsableUrl(rawUrl) {
  try {
    return EXTERNAL_PROTOCOLS.has(new URL(rawUrl).protocol);
  } catch {
    return false;
  }
}
