import { useTranslation } from "react-i18next";
import { ExternalLink } from "lucide-react";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SidebarNavLink } from "#/components/features/sidebar/sidebar-nav-link";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useCloudMcpSettingsUrl } from "#/hooks/use-cloud-mcp-settings-url";
import { cn } from "#/utils/utils";
import {
  SIDEBAR_ICON_SLOT_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavLabelClassName,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";
import { EXTENSIONS_NAV_ITEMS } from "./extensions-navigation";

/**
 * The Skills item points to a cloud-hosted page; so does the MCP Servers item
 * on an OpenHands Enterprise instance with organization MCP servers on.
 */
const CLOUD_LINKED_EXTENSION_PATH = "/skills";
/** Backend-installed artifacts are not available on Cloud backends yet. */
const CLOUD_HIDDEN_EXTENSION_PATHS = new Set(["/plugins", "/apps"]);

export function ExtensionsMobileHub() {
  const { t } = useTranslation("openhands");
  const { active } = useActiveBackendContext();
  const { backend } = active;
  const isCloudBackend = !isNoBackend(backend) && backend.kind === "cloud";
  const { url: cloudMcpSettingsUrl } = useCloudMcpSettingsUrl();

  return (
    <div
      data-testid="extensions-mobile-hub"
      className="flex flex-col gap-4 px-4 py-2 lg:hidden"
    >
      <Typography.H2>{t(I18nKey.NAV$CUSTOMIZE)}</Typography.H2>
      <nav className="flex flex-col gap-0.5">
        {EXTENSIONS_NAV_ITEMS.filter(
          (item) =>
            !(CLOUD_HIDDEN_EXTENSION_PATHS.has(item.to) && isCloudBackend),
        ).map((item) => {
          const isCloudSkillsLink =
            item.to === CLOUD_LINKED_EXTENSION_PATH && isCloudBackend;
          let externalUrl: string | null = null;
          if (isCloudSkillsLink) {
            externalUrl = `${backend.host.replace(/\/+$/, "")}/settings/skills`;
          } else if (item.to === "/mcp") {
            externalUrl = cloudMcpSettingsUrl;
          }
          if (externalUrl) {
            return (
              <a
                key={item.to}
                data-testid={`sidebar-extensions-${item.to}`}
                href={externalUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  sidebarNavRowClassName({ collapsed: false }),
                  SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
                )}
              >
                <span className={SIDEBAR_ICON_SLOT_CLASS}>{item.icon}</span>
                <span className={cn(sidebarNavLabelClassName(false), "flex-1")}>
                  {isCloudSkillsLink
                    ? t(I18nKey.SIDEBAR$SKILLS_AND_PLUGINS_CLOUD_LINK)
                    : item.label}
                </span>
                <ExternalLink
                  className="size-4 shrink-0 text-muted"
                  aria-hidden
                />
              </a>
            );
          }

          return (
            <SidebarNavLink
              key={item.to}
              to={item.to}
              label={item.label}
              end={item.end}
              testId={`sidebar-extensions-${item.to}`}
              icon={item.icon}
              disabled={item.comingSoon}
            />
          );
        })}
      </nav>
      <div className="pt-1">
        <BackendSyncedSettingsBadge />
      </div>
    </div>
  );
}
