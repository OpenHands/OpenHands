import { LOOPS_PATH } from "#/api/loop-service/loop-constants";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import type { SubPageNavItem } from "#/components/features/manifest/manifest-subpage-layout";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
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
 * manifest declares none. The templates item is hidden on cloud backends: the
 * catalog launcher it hosts is a local-backend feature and renders nothing
 * there. Loops are a local-only extension of the dashboard, not a sibling tab,
 * so the list item stays active while managing `/automations/loops`.
 */
export function useAutomationSubPageNav(): AutomationSubPageNav | null {
  const active = useActiveBackend();
  const { currentPath } = useNavigation();
  const spec = getSubPagesSpec();
  if (!spec) return null;

  const isCloudBackend = active.backend.kind === "cloud";
  const items: SubPageNavItem[] = spec
    .filter((item) => !(item.page === "templates" && isCloudBackend))
    .map((item) => ({
      to: item.to,
      label: item.label,
      // Templates is a catalog, so the host always shows the library icon
      // even while the published manifest still names sparkles.
      Icon:
        item.page === "templates"
          ? MANIFEST_ICON_BY_SLUG.library
          : MANIFEST_ICON_BY_SLUG[item.icon],
      testId: `automations-navigation-${item.page}`,
      forceActive: item.page === "list" && currentPath.startsWith(LOOPS_PATH),
    }));

  return {
    heading: getInterfaceCopy().sidebarLabel,
    items,
  };
}
