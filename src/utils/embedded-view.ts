export const EMBED_QUERY_PARAM = "embed";

/**
 * `?embed=1` renders a page without Canvas's own chrome (sidebar, menus,
 * extensions navigation) so another OpenHands surface can host it in a
 * same-origin frame, e.g. OpenHands Enterprise's classic MCP settings page.
 */
export function isEmbeddedView(
  search = typeof window !== "undefined" ? window.location.search : "",
): boolean {
  return new URLSearchParams(search).get(EMBED_QUERY_PARAM) === "1";
}
