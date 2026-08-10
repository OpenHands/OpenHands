import { ExternalLink } from "lucide-react";
import { getConfiguredSidebarLinks } from "#/config/sidebar-links";
import { NavigationLink } from "#/components/shared/navigation-link";
import { cn } from "#/utils/utils";
import { SIDEBAR_LINK_ICON_BY_SLUG } from "./sidebar-link-icons";
import {
  SIDEBAR_ICON_SLOT_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavLabelClassName,
  sidebarNavRowClassName,
} from "./sidebar-layout";

/** `to` when the URL targets this same deployment, `null` for a true external URL. */
function resolveInternalPath(url: string): string | null {
  if (typeof window === "undefined") return null;

  const parsed = new URL(url);
  if (parsed.origin !== window.location.origin) return null;

  return `${parsed.pathname}${parsed.search}${parsed.hash}`;
}

/**
 * Renders operator-configured links (see `#/config/sidebar-links`) after the
 * built-in nav items. Empty when no `AGENT_CANVAS_SIDEBAR_LINKS` config is
 * present, which is the common case.
 *
 * A configured URL that resolves to this same origin navigates in-app (no
 * new tab, no full reload) exactly like the built-in nav links above it —
 * an operator can point a link at an in-app route, not just an external
 * site. A genuinely external URL still opens in a new tab so it never
 * navigates the user's conversations away.
 */
export function SidebarConfiguredLinks({ collapsed }: { collapsed: boolean }) {
  const links = getConfiguredSidebarLinks();
  if (links.length === 0) return null;

  return (
    <>
      {links.map((link) => {
        const Icon = SIDEBAR_LINK_ICON_BY_SLUG[link.icon];
        const internalPath = resolveInternalPath(link.url);
        const rowClassName = cn(
          sidebarNavRowClassName({ collapsed }),
          SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
        );
        const content = (
          <>
            <span className={SIDEBAR_ICON_SLOT_CLASS}>
              <Icon className="size-[18px] shrink-0" aria-hidden />
            </span>
            <span className={cn(sidebarNavLabelClassName(collapsed), "flex-1")}>
              {link.label}
            </span>
            {!collapsed && internalPath === null && (
              <ExternalLink
                className="size-3.5 shrink-0 text-[var(--oh-muted)]"
                aria-hidden
              />
            )}
          </>
        );

        if (internalPath !== null) {
          return (
            <NavigationLink
              key={link.id}
              data-testid={`sidebar-configured-link-${link.id}`}
              to={internalPath}
              className={rowClassName}
            >
              {content}
            </NavigationLink>
          );
        }

        return (
          <a
            key={link.id}
            data-testid={`sidebar-configured-link-${link.id}`}
            href={link.url}
            target="_blank"
            rel="noopener noreferrer"
            className={rowClassName}
          >
            {content}
          </a>
        );
      })}
    </>
  );
}
