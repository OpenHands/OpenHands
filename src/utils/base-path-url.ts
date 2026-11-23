/**
 * React Router's `basename` ends with "/" because it must start with Vite's
 * `base`, so the bare base path ("/canvas") matches no route and renders a
 * blank page. Returns the same URL with the trailing slash added, or null when
 * the location needs no change.
 */
export function withBasePathSlash(
  location: Pick<Location, "pathname" | "search" | "hash">,
  base: string,
): string | null {
  if (base === "/" || !base.endsWith("/")) return null;
  if (location.pathname !== base.slice(0, -1)) return null;
  return `${base}${location.search}${location.hash}`;
}
