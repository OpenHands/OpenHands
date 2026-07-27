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

export function buildAgentCanvasPath(path: string): string {
  const basePath = getAgentCanvasBasePath();
  if (!basePath) return path.startsWith("/") ? path : `/${path}`;

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${basePath}${normalizedPath}`;
}
