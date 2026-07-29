import { useTranslation } from "react-i18next";
import { ExternalLink } from "lucide-react";
import { NavigationLink } from "#/components/shared/navigation-link";
import { cn } from "#/utils/utils";
import SkillsIcon from "#/icons/skills.svg?react";
import ServerProcessIcon from "#/icons/server-process.svg?react";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import {
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";

/** Only the Skills item points to a cloud-hosted page today. */
const CLOUD_LINKED_EXTENSION_PATH = "/skills";
/** Plugins are not available on Cloud backends, so this item is hidden there. */
const CLOUD_HIDDEN_EXTENSION_PATH = "/plugins";

interface ExtensionNavItem {
  to: string;
  label: string;
  icon: React.ReactElement;
  end?: boolean;
  comingSoon?: boolean;
}

export const EXTENSIONS_NAV_ITEMS: ExtensionNavItem[] = [
  {
    to: "/skills",
    label: "Skills",
    icon: <SkillsIcon width={16} height={16} aria-hidden="true" />,
    end: true,
  },
  {
    to: "/mcp",
    label: "MCP Servers",
    icon: <ServerProcessIcon width={16} height={16} />,
    end: true,
  },
  {
    to: "/plugins",
    label: "Plugins",
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
];

export function ExtensionsNavigation() {
  const { t } = useTranslation("openhands");
  const { active } = useActiveBackendContext();
  const { backend } = active;
  const isCloudBackend = !isNoBackend(backend) && backend.kind === "cloud";

  return (
    <aside
      data-testid="extensions-navbar-desktop"
      className="hidden md:flex md:w-[260px] md:shrink-0 md:flex-col md:gap-2 md:sticky md:top-8 md:self-start"
    >
      <span className="px-2 text-sm font-normal text-white">
        {t(I18nKey.NAV$CUSTOMIZE)}
      </span>
      <div className="flex flex-col gap-0.5 pt-0.5">
        {EXTENSIONS_NAV_ITEMS.filter(
          (item) =>
            !(item.to === CLOUD_HIDDEN_EXTENSION_PATH && isCloudBackend),
        ).map((item) => {
          const isCloudSkillsLink =
            item.to === CLOUD_LINKED_EXTENSION_PATH && isCloudBackend;
          const baseRow = (
            <span className="shrink-0 flex items-center justify-center">
              {item.icon}
            </span>
          );
          const label = (
            <span className="truncate">
              {isCloudSkillsLink
                ? t(I18nKey.SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK)
                : item.label}
            </span>
          );
          const comingSoonBadge = item.comingSoon && (
            <span className="ml-auto shrink-0 rounded-full border border-white/20 bg-white/5 px-1.5 py-0.5 text-[10px] font-medium text-[var(--oh-text-dim)]">
              {t(I18nKey.NAV$COMING_SOON)}
            </span>
          );

          if (isCloudSkillsLink) {
            const cloudSkillsUrl = `${backend.host.replace(/\/+$/, "")}/settings/skills`;
            return (
              <a
                key={item.to}
                data-testid={`sidebar-extensions-${item.to}`}
                href={cloudSkillsUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  sidebarNavRowClassName(),
                  "truncate",
                  SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
                )}
              >
                {baseRow}
                {label}
                <ExternalLink
                  className="ml-auto size-4 shrink-0 text-[var(--oh-muted)]"
                  aria-hidden
                />
              </a>
            );
          }

          return (
            <NavigationLink
              key={item.to}
              to={item.to}
              end={item.end}
              data-testid={`sidebar-extensions-${item.to}`}
              className={({ isActive }) =>
                cn(
                  sidebarNavRowClassName(),
                  "truncate",
                  isActive
                    ? SIDEBAR_ROW_INTERACTIVE_CLASS.active
                    : SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
                )
              }
            >
              {baseRow}
              {label}
              {comingSoonBadge}
            </NavigationLink>
          );
        })}
      </div>
      <div className="px-2 pt-3">
        <BackendSyncedSettingsBadge />
      </div>
    </aside>
  );
}
