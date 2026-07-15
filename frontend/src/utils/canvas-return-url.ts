export const AGENT_CANVAS_PATH_PREFIX = "/canvas";
export const AUTOMATIONS_PATH_PREFIX = "/automations";

const EXTERNAL_APP_PATH_PREFIXES = [
  AGENT_CANVAS_PATH_PREFIX,
  AUTOMATIONS_PATH_PREFIX,
] as const;

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

function isExternalAppPath(path: string): boolean {
  return EXTERNAL_APP_PATH_PREFIXES.some(
    (prefix) =>
      path === prefix ||
      path.startsWith(`${prefix}/`) ||
      path.startsWith(`${prefix}?`) ||
      path.startsWith(`${prefix}#`),
  );
}

export function getExternalAppReturnPath(value: string): string | null {
  const path = getSameOriginPath(value);
  if (!path) return null;

  return isExternalAppPath(path) ? path : null;
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

export function isCrossOriginUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.origin !== window.location.origin;
  } catch {
    return false;
  }
}

export function isAgentCanvasReturnUrl(path: string): boolean {
  return getAgentCanvasReturnPath(path) !== null;
}

export function navigateToReturnUrl(
  path: string,
  navigate: (path: string, options: { replace: true }) => void,
): void {
  const externalAppPath = getExternalAppReturnPath(path);
  if (externalAppPath) {
    window.location.assign(externalAppPath);
    return;
  }

  navigate(path, { replace: true });
}
