import React from "react";
import { useTranslation } from "react-i18next";
import { Cloud, Puzzle } from "lucide-react";
import { useNavigation } from "#/context/navigation-context";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { cloudIntegrationsUrl } from "#/utils/cloud-integrations-url";
import {
  TabletSiblingNav,
  TabletSiblingNavItem,
} from "#/components/shared/tablet-sibling-nav";
import { I18nKey } from "#/i18n/declaration";

export function TabletSettingsNav() {
  const { t } = useTranslation("openhands");
  const { currentPath } = useNavigation();
  const navItems = useSettingsNavItems();
  const { active } = useActiveBackendContext();
  const { backend, orgId } = active;

  if (currentPath === "/settings") {
    return null;
  }

  const isCloudBackend = !isNoBackend(backend) && backend.kind === "cloud";
  const isLockedToCloud = getLockedCloudHost() !== null;

  const items: TabletSiblingNavItem[] = [];

  for (const renderedItem of navItems) {
    if (renderedItem.type === "item") {
      items.push({
        to: renderedItem.item.to,
        label: t(renderedItem.item.text as I18nKey),
        icon: renderedItem.item.icon,
      });
    }
  }

  if (isCloudBackend) {
    const integrationsUrl = cloudIntegrationsUrl(backend, orgId);
    items.push({
      href: integrationsUrl,
      label: t(I18nKey.SETTINGS$INTEGRATIONS_SETTINGS_LINK),
      icon: <Puzzle className="size-4 shrink-0" aria-hidden />,
      isExternal: true,
      target: "_blank",
      rel: "noopener noreferrer",
    });

    const orgQuery = orgId ? `?org=${encodeURIComponent(orgId)}` : "";
    const cloudSettingsUrl = `${backend.host.replace(/\/+$/, "")}/settings${orgQuery}`;
    items.push({
      href: cloudSettingsUrl,
      label: t(I18nKey.SETTINGS$CLOUD_SETTINGS_LINK),
      icon: <Cloud className="size-4 shrink-0" aria-hidden />,
      isExternal: true,
      target: isLockedToCloud ? undefined : "_blank",
      rel: isLockedToCloud ? undefined : "noopener noreferrer",
    });
  }

  const currentItem = items.find((item) => item.to === currentPath);
  if (!currentItem) {
    return null;
  }

  const currentSectionLabel = currentItem.label;
  const currentSectionIcon = currentItem.icon;

  return (
    <TabletSiblingNav
      currentPath={currentPath}
      currentSectionLabel={currentSectionLabel}
      currentSectionIcon={currentSectionIcon}
      items={items}
      ariaLabel={t(I18nKey.NAV$SECTION_NAV_LABEL, {
        section: currentSectionLabel,
      })}
    />
  );
}
