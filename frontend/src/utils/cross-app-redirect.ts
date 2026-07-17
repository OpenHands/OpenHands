import type { NavigateFunction, NavigateOptions } from "react-router";

export function isCrossAppPath(destination: string): boolean {
  if (!destination.startsWith("/") || destination.startsWith("//")) {
    return false;
  }

  try {
    const { pathname } = new URL(destination, window.location.origin);
    return pathname === "/automations" || pathname.startsWith("/automations/");
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
