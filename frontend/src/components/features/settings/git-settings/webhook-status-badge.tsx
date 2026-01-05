import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export interface WebhookStatusBadgeProps {
  webhookInstalled: boolean;
  installationResult?: { success: boolean; error: string | null } | null;
}

export function WebhookStatusBadge({
  webhookInstalled,
  installationResult,
}: WebhookStatusBadgeProps) {
  const { t } = useTranslation();

  if (installationResult) {
    if (installationResult.success) {
      return (
        <Typography.Text className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
          {t(I18nKey.GITLAB$WEBHOOK_STATUS_INSTALLED)}
        </Typography.Text>
      );
    }
    return (
      <span title={installationResult.error || undefined}>
        <Typography.Text className="px-2 py-1 text-xs rounded bg-red-500/20 text-red-400">
          {t(I18nKey.GITLAB$WEBHOOK_STATUS_FAILED)}
        </Typography.Text>
      </span>
    );
  }

  if (webhookInstalled) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
        {t(I18nKey.GITLAB$WEBHOOK_STATUS_INSTALLED)}
      </Typography.Text>
    );
  }

  return (
    <Typography.Text className="px-2 py-1 text-xs rounded bg-gray-500/20 text-gray-400">
      {t(I18nKey.GITLAB$WEBHOOK_STATUS_NOT_INSTALLED)}
    </Typography.Text>
  );
}
