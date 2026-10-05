import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import type { SubPageNavItem } from "#/components/features/manifest/manifest-subpage-layout";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import {
  getInterfaceCopy,
  getSubPagesSpec,
} from "#/manifests/automation-interface";

export interface AutomationSubPageNav {
  heading: string;
  items: SubPageNavItem[];
}

/**
 * The manifest's sub-page navigation resolved for rendering, or null when the
 * manifest declares none.
 */
export function useAutomationSubPageNav(): AutomationSubPageNav | null {
  const { canManage } = useAutomationPermissions();
  const spec = getSubPagesSpec();
  if (!spec) return null;

  return {
    heading: getInterfaceCopy().sidebarLabel,
    items: spec
      .filter((item) => item.page !== "events" || canManage)
      .map((item) => ({
        to: item.to,
        label: item.label,
        Icon:
          item.page === "templates"
            ? MANIFEST_ICON_BY_SLUG.library
            : MANIFEST_ICON_BY_SLUG[item.icon],
        testId: `automations-navigation-${item.page}`,
      })),
  };
}
