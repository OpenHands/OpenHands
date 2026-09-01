import type { CanvasExtensionManifest } from "#/types/canvas-extension";

const SVG_EXTENSION = ".svg";

/**
 * Validate a manifest-declared icon path before it is ever turned into a URL.
 *
 * The path must stay inside the extension's installed root:
 *  - relative only (no leading slash, drive letter, or URL scheme),
 *  - no parent-directory (`..`) segments, backslashes, or percent-encoded
 *    variants of either,
 *  - an `.svg` extension so the asset is rendered as an image (SVGs loaded
 *    via `<img>` cannot execute scripts).
 *
 * The agent-server enforces the same rules and serves only files it can
 * confirm are SVG; this frontend check exists so an invalid path degrades to
 * the default icon instead of hitting the network.
 */
export function isValidCanvasExtensionIconPath(iconPath: string): boolean {
  const trimmed = iconPath.trim();
  if (trimmed.length === 0) return false;

  // Backslashes are never valid in an extension-relative path and are the
  // classic Windows traversal vehicle, so reject them outright.
  if (trimmed.includes("\\")) return false;

  // Must be relative: no leading slash, drive letter, or URL scheme.
  if (trimmed.startsWith("/")) return false;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)) return false;

  // No parent-directory traversal, plain or percent-encoded.
  if (trimmed.split("/").some((segment) => segment === "..")) return false;
  if (trimmed.includes("%")) return false;

  return trimmed.toLowerCase().endsWith(SVG_EXTENSION);
}

/**
 * Resolve the custom SVG icon declared by an extension manifest, or `null`
 * when the manifest has no icon or the declared path is unsafe/invalid.
 */
export function getCanvasExtensionIconPath(
  manifest: CanvasExtensionManifest | null | undefined,
): string | null {
  const icon = manifest?.icon;
  if (typeof icon !== "string") return null;
  const trimmed = icon.trim();
  return isValidCanvasExtensionIconPath(trimmed) ? trimmed : null;
}
