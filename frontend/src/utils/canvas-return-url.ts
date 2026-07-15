export const AGENT_CANVAS_PATH_PREFIX = "/canvas";

export function isAgentCanvasReturnUrl(path: string): boolean {
  return (
    path === AGENT_CANVAS_PATH_PREFIX ||
    path.startsWith(`${AGENT_CANVAS_PATH_PREFIX}/`)
  );
}

export function navigateToReturnUrl(
  path: string,
  navigate: (path: string, options: { replace: true }) => void,
): void {
  if (isAgentCanvasReturnUrl(path)) {
    window.location.assign(path);
    return;
  }

  navigate(path, { replace: true });
}
