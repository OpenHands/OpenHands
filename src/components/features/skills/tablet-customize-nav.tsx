import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";
import {
  EXTENSIONS_NAV_ITEMS,
  CLOUD_HIDDEN_EXTENSION_PATHS,
  CLOUD_LINKED_EXTENSION_PATH,
} from "#/components/features/skills/extensions-navigation";
import {
  TabletSiblingNav,
  TabletSiblingNavItem,
} from "#/components/shared/tablet-sibling-nav";
import { I18nKey } from "#/i18n/declaration";

export function TabletCustomizeNav() {
  const { t } = useTranslation("openhands");
  const { currentPath } = useNavigation();
  const { active } = useActiveBackendContext();
  const { backend } = active;
  const isCloudBackend = !isNoBackend(backend) && backend.kind === "cloud";

  if (currentPath === "/customize") {
    return null;
  }

  const items: TabletSiblingNavItem[] = EXTENSIONS_NAV_ITEMS.filter(
    (item) => !(CLOUD_HIDDEN_EXTENSION_PATHS.has(item.to) && isCloudBackend),
  ).map((item) => {
    const isCloudSkillsLink =
      item.to === CLOUD_LINKED_EXTENSION_PATH && isCloudBackend;
    if (isCloudSkillsLink) {
      const cloudSkillsUrl = `${backend.host.replace(/\/+$/, "")}/settings/skills`;
      return {
        href: cloudSkillsUrl,
        label: t(I18nKey.SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK),
        icon: item.icon,
        isExternal: true,
        target: "_blank",
        rel: "noopener noreferrer",
      };
    }
    return {
      to: item.to,
      label: item.label,
      icon: item.icon,
    };
  });

  const currentItem = items.find((item) => item.to === currentPath);
  const currentSectionLabel = currentItem?.label ?? t(I18nKey.NAV$CUSTOMIZE);
  const currentSectionIcon = currentItem?.icon;

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
