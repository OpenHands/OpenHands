import React from "react";
import { redirect, useNavigate } from "react-router";
import {
  isExtensionHomePath,
  readPinnedHomeRoute,
  usePinnedHomeRoute,
} from "#/hooks/use-pinned-home-route";
import { useCanvasExtensionsRuntime } from "#/components/features/canvas-extensions/canvas-extensions-runtime";
import HomeComponent from "./home";

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

/**
 * Canvas Extension pins redirect here rather than in the loader: whether the
 * pinned page exists is only known once the extension runtime resolves, and
 * a disabled or uninstalled extension must fall back to the default home
 * with no error and no redirect to the unavailable extension (#16995).
 */
function PinnedExtensionHomeRedirect() {
  const { pages } = useCanvasExtensionsRuntime();
  const extensionHrefs = React.useMemo(
    () => new Set(pages.map((page) => page.href)),
    [pages],
  );
  const { pinnedRoute } = usePinnedHomeRoute(extensionHrefs);
  const navigate = useNavigate();

  React.useEffect(() => {
    if (!pinnedRoute || !isExtensionHomePath(pinnedRoute)) return;
    if (pages.some((page) => page.href === pinnedRoute)) {
      navigate(pinnedRoute, { replace: true });
    }
  }, [pinnedRoute, pages, navigate]);

  return null;
}

export default function IndexHome() {
  return (
    <>
      <PinnedExtensionHomeRedirect />
      <HomeComponent />
    </>
  );
}
