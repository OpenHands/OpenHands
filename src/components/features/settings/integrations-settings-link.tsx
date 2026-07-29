import { useTranslation } from "react-i18next";
import { ExternalLink, Puzzle } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { cn } from "#/utils/utils";
import {
  SIDEBAR_ICON_SLOT_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavLabelClassName,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";

/**
 * Renders only for cloud backends — local backends have no equivalent
 * hosted integrations settings page.
 */
export function IntegrationsSettingsLink() {
  const { t } = useTranslation("openhands");
  const { active } = useActiveBackendContext();
  const { backend } = active;

  if (isNoBackend(backend) || backend.kind !== "cloud") return null;

  const integrationsUrl = `${backend.host.replace(/\/+$/, "")}/settings/integrations`;

  return (
    <a
      data-testid="settings-integrations-link"
      href={integrationsUrl}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        sidebarNavRowClassName({ collapsed: false }),
        SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
      )}
    >
      <span className={SIDEBAR_ICON_SLOT_CLASS}>
        <Puzzle className="size-4 shrink-0" aria-hidden />
      </span>
      <span className={cn(sidebarNavLabelClassName(false), "flex-1")}>
        {t(I18nKey.SETTINGS$INTEGRATIONS_SETTINGS_LINK)}
      </span>
      <ExternalLink
        className="size-4 shrink-0 text-[var(--oh-muted)]"
        aria-hidden
      />
    </a>
  );
}
