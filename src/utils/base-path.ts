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
 */
function stripBasePath(origin: string): string {
  const basePath = getAgentCanvasBasePath();
  if (!basePath || !origin.endsWith(basePath)) return origin;
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
