import React from "react";
import { useTranslation } from "react-i18next";
import { Blocks, ExternalLink } from "lucide-react";
import { CompactSectionNavigation } from "#/components/shared/compact-section-navigation";
import { useNavigation } from "#/context/navigation-context";
import { NavigationLink } from "#/components/shared/navigation-link";
import { cn } from "#/utils/utils";
import SkillsIcon from "#/icons/skills.svg?react";
import ServerProcessIcon from "#/icons/server-process.svg?react";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import {
  SIDEBAR_NAV_ROW_WITH_DESCRIPTION_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";
import { SidebarNavText } from "#/components/features/sidebar/sidebar-nav-link";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";

/** Only the Skills item points to a cloud-hosted page today. */
const CLOUD_LINKED_EXTENSION_PATH = "/skills";
/** Backend-installed artifacts are not available on Cloud backends yet. */
const CLOUD_HIDDEN_EXTENSION_PATHS = new Set(["/plugins", "/apps"]);

interface ExtensionNavItem {
  to: string;
  label: I18nKey;
  /** Plain-language explanation of the section for first-time users. */
  description: I18nKey;
  icon: React.ReactElement;
  end?: boolean;
  comingSoon?: boolean;
}

export const EXTENSIONS_NAV_ITEMS: ExtensionNavItem[] = [
  {
    to: "/mcp",
    label: I18nKey.NAV$CUSTOMIZE_MCP_SERVERS,
    description: I18nKey.NAV$CUSTOMIZE_MCP_DESCRIPTION,
    icon: <ServerProcessIcon width={16} height={16} />,
    end: true,
  },
  {
    to: "/skills",
    label: I18nKey.SETTINGS$SKILLS_TITLE,
    description: I18nKey.NAV$CUSTOMIZE_SKILLS_DESCRIPTION,
    icon: <SkillsIcon width={16} height={16} aria-hidden="true" />,
    end: true,
  },
  {
    to: "/plugins",
    label: I18nKey.SETTINGS$PLUGINS_TITLE,
    description: I18nKey.NAV$CUSTOMIZE_PLUGINS_DESCRIPTION,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        width={16}
        height={16}
        aria-hidden="true"
      >
        <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" />
        <path d="m3.3 7 8.7 5 8.7-5" />
        <path d="M12 22V12" />
      </svg>
    ),
    end: true,
  },
  {
    to: "/apps",
    label: I18nKey.NAV$EXTENSIONS,
    description: I18nKey.NAV$CUSTOMIZE_APPS_DESCRIPTION,
    icon: <Blocks width={16} height={16} aria-hidden="true" />,
    end: true,
  },
];

export function ExtensionsNavigation() {
  const { t } = useTranslation("openhands");

  return (
    <aside
      data-testid="extensions-navbar-desktop"
      className="hidden lg:flex lg:w-65 lg:shrink-0 lg:flex-col lg:gap-2 lg:sticky lg:top-8 lg:self-start"
    >
      <span className="px-2 text-sm font-normal text-contrast">
        {t(I18nKey.NAV$CUSTOMIZE)}
      </span>
      <ExtensionsNavigationLinks />
      <div className="px-2 pt-3">
        <BackendSyncedSettingsBadge />
      </div>
    </aside>
  );
}

function ExtensionsNavigationLinks() {
  const { active } = useActiveBackendContext();
  const { backend } = active;
  const isCloudBackend = !isNoBackend(backend) && backend.kind === "cloud";
  return (
    <div className="flex flex-col gap-0.5 pt-0.5">
      {EXTENSIONS_NAV_ITEMS.filter(
        (item) =>
          !(CLOUD_HIDDEN_EXTENSION_PATHS.has(item.to) && isCloudBackend),
      ).map((item) => (
        <ExtensionsNavigationRow
          key={item.to}
          item={item}
          cloudSkillsHost={
            item.to === CLOUD_LINKED_EXTENSION_PATH && isCloudBackend
              ? backend.host
              : null
          }
        />
      ))}
    </div>
  );
}

function ExtensionsNavigationRow({
  item,
  cloudSkillsHost,
}: {
  item: ExtensionNavItem;
  /** Set when the row links out to the Cloud-hosted Skills page instead. */
  cloudSkillsHost: string | null;
}) {
  const { t } = useTranslation("openhands");
  const labelId = React.useId();
  const descriptionId = React.useId();
  const icon = (
    <span className="shrink-0 flex items-center justify-center">
      {item.icon}
    </span>
  );
  const text = (
    <SidebarNavText
      label={t(
        cloudSkillsHost === null
          ? item.label
          : I18nKey.SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK,
      )}
      description={t(item.description)}
      labelId={labelId}
      descriptionId={descriptionId}
    />
  );
  const rowClassName = cn(
    sidebarNavRowClassName(),
    SIDEBAR_NAV_ROW_WITH_DESCRIPTION_CLASS,
  );

  if (cloudSkillsHost !== null) {
    const cloudSkillsUrl = `${cloudSkillsHost.replace(/\/+$/, "")}/settings/skills`;
    return (
      <a
        data-testid={`sidebar-extensions-${item.to}`}
        href={cloudSkillsUrl}
        target="_blank"
        rel="noopener noreferrer"
        aria-labelledby={labelId}
        aria-describedby={descriptionId}
        className={cn(rowClassName, SIDEBAR_ROW_INTERACTIVE_CLASS.idle)}
      >
        {icon}
        {text}
        <ExternalLink
          className="ml-auto size-4 shrink-0 text-muted"
          aria-hidden
        />
      </a>
    );
  }

  return (
    <NavigationLink
      to={item.to}
      end={item.end}
      data-testid={`sidebar-extensions-${item.to}`}
      aria-labelledby={labelId}
      aria-describedby={descriptionId}
      className={({ isActive }) =>
        cn(
          rowClassName,
          isActive
            ? SIDEBAR_ROW_INTERACTIVE_CLASS.active
            : SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
        )
      }
    >
      {icon}
      {text}
      {item.comingSoon && (
        <span className="ml-auto shrink-0 rounded-full border border-contrast/20 bg-contrast/5 px-1.5 py-0.5 text-[10px] font-medium text-text-dim">
          {t(I18nKey.NAV$COMING_SOON)}
        </span>
      )}
    </NavigationLink>
  );
}

export function ExtensionsCompactNavigation() {
  const { t } = useTranslation("openhands");
  const { currentPath } = useNavigation();
  const currentItem = EXTENSIONS_NAV_ITEMS.find(
    (item) => currentPath === item.to || currentPath.startsWith(`${item.to}/`),
  );

  return (
    <CompactSectionNavigation
      label={t(I18nKey.NAV$CUSTOMIZE)}
      currentLabel={currentItem ? t(currentItem.label) : undefined}
      testId="extensions-compact-navigation"
    >
      <ExtensionsNavigationLinks />
    </CompactSectionNavigation>
  );
}
