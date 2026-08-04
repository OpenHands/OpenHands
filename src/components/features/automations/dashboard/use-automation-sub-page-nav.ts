import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import type { SubPageNavItem } from "#/components/features/manifest/manifest-subpage-layout";
import { useActiveBackend } from "#/contexts/active-backend-context";
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
 * there.
 */
export function useAutomationSubPageNav(): AutomationSubPageNav | null {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const spec = getSubPagesSpec();
  if (!spec) return null;

  const isCloudBackend = active.backend.kind === "cloud";
  return {
    heading: getInterfaceCopy().sidebarLabel ?? t(I18nKey.SIDEBAR$AUTOMATIONS),
    items: spec
      .filter((item) => !(item.page === "templates" && isCloudBackend))
      .map((item) => ({
        to: item.to,
        label: item.label,
        Icon: MANIFEST_ICON_BY_SLUG[item.icon],
        testId: `automations-navigation-${item.page}`,
      })),
  };
}
