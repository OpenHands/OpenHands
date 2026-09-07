import { useTranslation } from "react-i18next";
import { PanelsTopLeft } from "lucide-react";
import { SidebarNavLink } from "./sidebar-nav-link";
import { RuckusNavIcon } from "./ruckus-nav-icon";
import { useNavigation } from "#/context/navigation-context";
import { useCanvasExtensionsRuntime } from "#/components/features/canvas-extensions/canvas-extensions-runtime";
import {
  CUSTOMIZE_PATH,
  usePinnedHomeRoute,
} from "#/hooks/use-pinned-home-route";
import {
  automationListPath,
  getInterfaceCopy,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import { I18nKey } from "#/i18n/declaration";

export function RuckusPrimaryNavigation({
  collapsed = false,
  includeConversations = true,
}: {
  collapsed?: boolean;
  includeConversations?: boolean;
}) {
  const { t } = useTranslation("openhands");
  const { currentPath } = useNavigation();
  const { pages: canvasExtensionPages } = useCanvasExtensionsRuntime();
  const { isPinnedRoute, togglePinnedRoute } = usePinnedHomeRoute();
  const isExtensionsActive =
    currentPath === CUSTOMIZE_PATH ||
    ["/skills", "/plugins", "/extensions", "/mcp"].some(
      (path) => currentPath === path || currentPath.startsWith(`${path}/`),
    );
  const buildPinAction = (path: string, testId: string) => {
    const pinned = isPinnedRoute(path);
    return {
      pinned,
      onToggle: () => togglePinnedRoute(path),
      label: t(
        pinned ? I18nKey.SIDEBAR$UNPIN_AS_HOME : I18nKey.SIDEBAR$PIN_AS_HOME,
      ),
      testId,
    };
  };
  return (
    <>
      {includeConversations && (
        <SidebarNavLink
          to="/conversations"
          label={t(I18nKey.SIDEBAR$CONVERSATIONS)}
          collapsed={collapsed}
          forceActive={currentPath === "/"}
          icon={<RuckusNavIcon kind="conversations" />}
        />
      )}
      {/* The interface manifest owns this entry's label, so an absent
            manifest leaves the rail without it rather than with host copy. */}
      {hasAutomationInterface() && (
        <SidebarNavLink
          to={automationListPath()}
          label={getInterfaceCopy().sidebarLabel}
          testId="sidebar-automations-link"
          collapsed={collapsed}
          icon={<RuckusNavIcon kind="automations" />}
          pinAction={buildPinAction(
            automationListPath(),
            "sidebar-pin-home-toggle-automations",
          )}
        />
      )}
      <SidebarNavLink
        to={CUSTOMIZE_PATH}
        label={t(I18nKey.NAV$CUSTOMIZE)}
        testId="sidebar-skills-link"
        collapsed={collapsed}
        forceActive={isExtensionsActive}
        pinAction={buildPinAction(
          CUSTOMIZE_PATH,
          "sidebar-pin-home-toggle-customize",
        )}
        icon={<RuckusNavIcon kind="customize" />}
      />
      {canvasExtensionPages.map((page) => (
        <SidebarNavLink
          key={`${page.extension.name}:${page.contribution.id}`}
          to={page.href}
          label={page.contribution.nav_label || page.contribution.title}
          testId={`sidebar-canvas-extension-${page.extension.name}-${page.contribution.id}`}
          collapsed={collapsed}
          icon={<PanelsTopLeft width={18} height={18} />}
        />
      ))}
    </>
  );
}
