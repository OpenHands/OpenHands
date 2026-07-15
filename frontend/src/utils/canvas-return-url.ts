export const AGENT_CANVAS_PATH_PREFIX = "/canvas";

function getSameOriginPath(value: string): string | null {
  try {
    const isRelativePath = /^\/(?!\/)/.test(value);
    const base =
      typeof window === "undefined"
        ? "http://localhost"
        : window.location.origin;
    const url = new URL(value, base);

    if (!isRelativePath && url.origin !== base) {
      return null;
    }

    return `${url.pathname}${url.search}${url.hash}`;
  } catch {
    return null;
  }
}

export function getAgentCanvasReturnPath(value: string): string | null {
  const path = getSameOriginPath(value);
  if (!path) return null;

  if (
    path === AGENT_CANVAS_PATH_PREFIX ||
    path.startsWith(`${AGENT_CANVAS_PATH_PREFIX}/`) ||
    path.startsWith(`${AGENT_CANVAS_PATH_PREFIX}?`) ||
    path.startsWith(`${AGENT_CANVAS_PATH_PREFIX}#`)
  ) {
    return path;
  }

  return null;
}

export function isAgentCanvasReturnUrl(path: string): boolean {
  return getAgentCanvasReturnPath(path) !== null;
}

export function navigateToReturnUrl(
  path: string,
  navigate: (path: string, options: { replace: true }) => void,
): void {
  const canvasPath = getAgentCanvasReturnPath(path);
  if (canvasPath) {
    window.location.assign(canvasPath);
    return;
  }

  navigate(path, { replace: true });
}
