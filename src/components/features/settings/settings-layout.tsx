import { SettingsDesktopSidebar } from "./settings-desktop-sidebar";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { settingsLayoutMainScrollClassName } from "#/utils/settings-like-page-layout-classes";

interface SettingsLayoutProps {
  children: React.ReactNode;
  navigationItems: SettingsNavRenderedItem[];
}

/**
 * Mirrors the extensions layout (Skills / MCP): aside and main are siblings,
 * and only the main column scrolls so the left nav stays pinned like
 * ExtensionsNavigation.
 */
export function SettingsLayout({
  children,
  navigationItems,
}: SettingsLayoutProps) {
  return (
    <div className="flex h-full flex-col md:pt-8">
      <div className="flex min-h-0 flex-1 gap-10 md:items-start">
        <SettingsDesktopSidebar navigationItems={navigationItems} />
        <main className={settingsLayoutMainScrollClassName}>
          <div className="mx-auto w-full min-w-0 max-w-[800px]">{children}</div>
        </main>
      </div>
    </div>
  );
}
