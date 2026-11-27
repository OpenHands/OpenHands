import { redirect } from "react-router";
import { readPinnedHomeRoute } from "#/hooks/use-pinned-home-route";

/**
 * With a home pin set for the active backend + org, `/` redirects to the
 * pinned page. `readPinnedHomeRoute` only returns routes that currently
 * resolve (never `/` itself), so a stale pin falls back to the default
 * home with no error and no redirect loop; the built-in home stays
 * reachable unmodified at /conversations.
 */
export const clientLoader = () => {
  const pinnedRoute = readPinnedHomeRoute();
  if (pinnedRoute) return redirect(pinnedRoute);
  return null;
};

export { default } from "./home";
