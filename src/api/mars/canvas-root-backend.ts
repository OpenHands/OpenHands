/**
 * The Canvas server a page came from, and the saved backend that holds its
 * key.
 *
 * Bridge calls and pairing go to the page's own origin, so only a key saved
 * for that origin belongs there. That rules out a backend the build points
 * elsewhere (`VITE_BACKEND_BASE_URL`) and a DigitalOcean session's backend,
 * which is also `kind: "local"` on this origin but under
 * `/mars/sessions/<id>`.
 */

import type { Backend } from "#/api/backend-registry/types";

/** Whether `url` is this page's origin itself, with no path. */
export function isPageOrigin(url: string | null | undefined): boolean {
  if (!url || typeof window === "undefined") return false;
  try {
    const parsed = new URL(url.trim());
    return (
      parsed.origin === window.location.origin &&
      (parsed.pathname === "/" || parsed.pathname === "")
    );
  } catch {
    return false;
  }
}

/** The backend a typed or paired Canvas key is saved on. */
export function isCanvasRootBackend(backend: Backend): boolean {
  return (
    backend.kind === "local" &&
    !backend.marsSessionId &&
    isPageOrigin(backend.host)
  );
}
