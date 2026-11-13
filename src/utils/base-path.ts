const BASE_PATH_WINDOW_KEY = "__AGENT_CANVAS_BASE_PATH__";

function normalizeBasePath(value?: string | null): string {
  const raw = value?.trim();
  if (!raw || raw === "/") return "";

  const withLeadingSlash = raw.startsWith("/") ? raw : `/${raw}`;
  return withLeadingSlash.replace(/\/+$/, "");
}

export function getAgentCanvasBasePath(): string {
  const envPath = normalizeBasePath(import.meta.env.VITE_BASE_PATH);
  if (envPath) return envPath;

  if (typeof window !== "undefined") {
    const injected = (window as unknown as Record<string, unknown>)[
      BASE_PATH_WINDOW_KEY
    ];
    if (typeof injected === "string") {
      return normalizeBasePath(injected);
    }
  }

  return "";
}

/**
 * Whether `path` already carries its own origin (an absolute URL or a
 * protocol-relative one). Such paths address another document and must never
 * have the Canvas base path spliced into them.
 */
function hasOwnOrigin(path: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("//");
}

/**
 * Prefix an app-relative path with the Canvas base path.
 *
 * Canvas may be mounted under a subpath (e.g. `/canvas`) on a host that also
 * serves other apps, so any URL that leaves the app — an `href` the browser
 * follows itself, a share link, an exported record — must carry the base path.
 * Bare paths resolve against the origin root and land on whatever other app
 * owns that path.
 */
export function buildAgentCanvasPath(path: string): string {
  if (hasOwnOrigin(path)) return path;

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const basePath = getAgentCanvasBasePath();
  if (!basePath) return normalizedPath;

  return `${basePath}${normalizedPath}`;
}

function resolveOrigin(origin?: string): string {
  if (origin !== undefined) return origin.replace(/\/+$/, "");
  return typeof window !== "undefined" ? window.location.origin : "";
}

/**
 * Drop a trailing base path so an origin that already names the Canvas mount
 * point (`https://host/canvas`) is not prefixed with it a second time.
 *
 * The match is limited to the URL path: an origin whose hostname merely
 * matches the mount segment (`https://canvas` with base path `/canvas`) has
 * no path at all, so nothing is stripped and the base path is applied once
 * to the hostname instead of being spliced into it.
 */
function stripBasePath(origin: string): string {
  const basePath = getAgentCanvasBasePath();
  if (!basePath) return origin;

  // Locate the start of the path, i.e. the first slash after the authority
  // (`scheme://host[:port]` or `//host[:port]`). An origin with no such slash
  // has no path, so a hostname that looks like the base path is not a match.
  const schemeIndex = origin.indexOf("://");
  const authorityEnd =
    schemeIndex === -1 ? (origin.startsWith("//") ? 2 : 0) : schemeIndex + 3;
  const pathStart = origin.indexOf("/", authorityEnd);
  if (pathStart === -1) return origin;

  const path = origin.slice(pathStart);
  // `basePath` begins with `/`, so an endsWith match is segment-aligned on
  // its own: `/foo/canvas` ends with `/canvas` but `/mycanvas` does not. The
  // previous adjacency check could never hold for a multi-segment path —
  // the character before the match is the last character of the preceding
  // segment, not `/` — so origins already carrying the mount under a deeper
  // path were double-prefixed instead of stripped once.
  if (!path.endsWith(basePath)) return origin;
  return origin.slice(0, -basePath.length);
}

/**
 * The absolute URL of the Canvas app root — origin plus base path, with no
 * trailing slash (`https://host/canvas`). Use this as the base for records that
 * append their own app paths.
 */
export function getAgentCanvasBaseUrl(origin?: string): string {
  return `${stripBasePath(resolveOrigin(origin))}${getAgentCanvasBasePath()}`;
}

/**
 * Build an absolute, base-path-aware URL for an app-relative path.
 *
 * `origin` defaults to the current document's origin and may itself include the
 * base path, which is not then applied twice.
 */
export function buildAgentCanvasUrl(path: string, origin?: string): string {
  if (hasOwnOrigin(path)) return path;

  const originRoot = stripBasePath(resolveOrigin(origin));
  if (!originRoot) return buildAgentCanvasPath(path);

  return `${originRoot}${buildAgentCanvasPath(path)}`;
}
