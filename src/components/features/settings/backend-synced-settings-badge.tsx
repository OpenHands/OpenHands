import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAllCloudOrganizations } from "#/hooks/query/use-cloud-organizations";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";

function useActiveBackendDisplayName(): string {
  const { t } = useTranslation();
  const active = useActiveBackend();
  const cloudOrgs = useAllCloudOrganizations();
  const userIds = useCloudCurrentUserId();

  if (active.backend.kind !== "cloud" || !active.orgId) {
    return active.backend.name;
  }

  const entry = cloudOrgs[active.backend.id];
  const org = entry?.orgs.find((o) => o.id === active.orgId);
  if (!org) return active.backend.name;

  const userId = userIds[active.backend.id]?.userId ?? null;
  const isPersonal = !!userId && org.id === userId;
  const orgLabel = isPersonal
    ? t(I18nKey.BACKEND$PERSONAL_WORKSPACE)
    : org.name;
  return `${active.backend.name} – ${orgLabel}`;
}

export function BackendSyncedSettingsBadge() {
  const { t } = useTranslation();
  const active = useActiveBackend();
  const name = useActiveBackendDisplayName();

  return (
    <div
      data-testid="backend-synced-settings-badge"
      className="flex items-center"
    >
      <Typography.Text className="text-[11px] font-medium text-[var(--oh-text-dim)] leading-5">
        {t(I18nKey.SETTINGS$BACKEND_SYNCED_BADGE, {
          name,
          host: active.backend.host,
          interpolation: { escapeValue: false },
        })}
      </Typography.Text>
    </div>
  );
}
