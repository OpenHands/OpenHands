import type { NavigateFunction, NavigateOptions } from "react-router";

const CROSS_APP_PATH_PREFIXES = ["/automations", "/canvas"] as const;

export function isCrossAppPath(destination: string): boolean {
  if (!destination.startsWith("/") || destination.startsWith("//")) {
    return false;
  }

  try {
    const { pathname } = new URL(destination, window.location.origin);
    return CROSS_APP_PATH_PREFIXES.some(
      (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
    );
  } catch {
    return false;
  }
}

export function navigateOrHardRedirect(
  navigate: NavigateFunction,
  destination: string,
  options?: NavigateOptions,
) {
  if (isCrossAppPath(destination)) {
    window.location.replace(destination);
    return;
  }

  navigate(destination, options);
}
