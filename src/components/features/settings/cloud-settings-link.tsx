import { useTranslation } from "react-i18next";
import { Cloud, ExternalLink } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { cn } from "#/utils/utils";
import {
  SIDEBAR_ICON_SLOT_CLASS,
  SIDEBAR_ROW_INTERACTIVE_CLASS,
  sidebarNavLabelClassName,
  sidebarNavRowClassName,
} from "#/components/features/sidebar/sidebar-layout";

/**
 * Renders only for cloud backends — local backends have no equivalent
 * hosted settings page. Opens in the same tab when the canvas is locked to a
 * Cloud host (SaaS / self-hosted OHE): the settings page shares that host
 * with the canvas, so Back should return here (OHE-3242).
 */
export function CloudSettingsLink() {
  const { t } = useTranslation("openhands");
  const { active } = useActiveBackendContext();
  const { backend, orgId } = active;

  if (isNoBackend(backend) || backend.kind !== "cloud") return null;

  const isLockedToCloud = getLockedCloudHost() !== null;

  // `org` is consumed by the cloud settings loader so the page opens on the
  // org that is active here instead of the cloud's last-used org.
  const orgQuery = orgId ? `?org=${encodeURIComponent(orgId)}` : "";
  const cloudSettingsUrl = `${backend.host.replace(/\/+$/, "")}/settings${orgQuery}`;

  return (
    <a
      data-testid="settings-cloud-link"
      href={cloudSettingsUrl}
      target={isLockedToCloud ? undefined : "_blank"}
      rel={isLockedToCloud ? undefined : "noopener noreferrer"}
      className={cn(
        sidebarNavRowClassName({ collapsed: false }),
        SIDEBAR_ROW_INTERACTIVE_CLASS.idle,
      )}
    >
      <span className={SIDEBAR_ICON_SLOT_CLASS}>
        <Cloud className="size-4 shrink-0" aria-hidden />
      </span>
      <span className={cn(sidebarNavLabelClassName(false), "flex-1")}>
        {t(I18nKey.SETTINGS$CLOUD_SETTINGS_LINK)}
      </span>
      {!isLockedToCloud && (
        <ExternalLink
          className="size-4 shrink-0 text-[var(--oh-muted)]"
          aria-hidden
        />
      )}
    </a>
  );
}
