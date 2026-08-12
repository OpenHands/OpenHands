import type { LucideIcon } from "lucide-react";
import { NavigationLink } from "#/components/shared/navigation-link";
import {
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import { cn } from "#/utils/utils";

export interface SubPageNavItem {
  to: string;
  label: string;
  Icon: LucideIcon;
  testId: string;
}

interface ManifestSubpageLayoutProps {
  /** The section heading above the desktop navigation. */
  heading: string;
  /** Prefix for the desktop/mobile nav testids. */
  navTestIdBase: string;
  items: SubPageNavItem[];
  children: React.ReactNode;
}

function SubPageNavLink({ item }: { item: SubPageNavItem }) {
  return (
    <NavigationLink
      to={item.to}
      end
      data-testid={item.testId}
      className={({ isActive }) =>
        cn(
          sidebarNavRowClassName(),
          "truncate whitespace-nowrap",
          isActive
            ? SIDEBAR_ROW_INTERACTIVE_CLASS.active
            : SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
        )
      }
    >
      <span className="shrink-0 flex items-center justify-center">
        <item.Icon size={16} aria-hidden />
      </span>
      <span className="truncate">{item.label}</span>
    </NavigationLink>
  );
}

/**
 * The shell of a manifest-declared sub-page: a sticky desktop aside and a
 * horizontal mobile strip around a settings-like scrolling content column.
 * Purely presentational — what the items are, and whether any exist at all,
 * is the caller's (ultimately the manifest's) statement.
 */
export function ManifestSubpageLayout({
  heading,
  navTestIdBase,
  items,
  children,
}: ManifestSubpageLayoutProps) {
  return (
    <div className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10">
      <aside
        data-testid={`${navTestIdBase}-desktop`}
        className="hidden md:flex md:w-[260px] md:shrink-0 md:flex-col md:gap-2 md:sticky md:top-8 md:self-start"
      >
        <span className="px-2 text-sm font-normal text-white">{heading}</span>
        <div className="flex flex-col gap-0.5 pt-0.5">
          {items.map((item) => (
            <SubPageNavLink key={item.to} item={item} />
          ))}
        </div>
      </aside>
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
          <nav
            data-testid={`${navTestIdBase}-mobile`}
            className="md:hidden flex gap-1 overflow-x-auto border-b border-[var(--oh-border)] pb-2"
          >
            {items.map((item) => (
              <SubPageNavLink key={item.to} item={item} />
            ))}
          </nav>
          {children}
        </div>
      </main>
    </div>
  );
}
