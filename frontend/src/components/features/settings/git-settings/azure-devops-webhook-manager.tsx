import { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { AzureDevOpsWebhookStatus } from "#/api/integration-service/integration-service.types";
import { useAzureDevOpsResources } from "#/hooks/query/use-azure-devops-resources-list";
import { useReinstallAzureDevOpsWebhook } from "#/hooks/mutation/use-reinstall-azure-devops-webhook";
import { useUninstallAzureDevOpsWebhook } from "#/hooks/mutation/use-uninstall-azure-devops-webhook";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface AzureDevOpsWebhookManagerProps {
  className?: string;
}

function StatusBadge({ status }: { status: AzureDevOpsWebhookStatus }) {
  const { t } = useTranslation();

  if (!status.webhook_secret_set) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-red-500/20 text-red-400">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_STATUS_MISSING_SECRET)}
      </Typography.Text>
    );
  }

  if (status.webhook_installed) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_STATUS_INSTALLED)}
      </Typography.Text>
    );
  }

  if (status.pr_webhook_installed || status.work_item_webhook_installed) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-yellow-500/20 text-yellow-300">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_STATUS_PARTIAL)}
      </Typography.Text>
    );
  }

  return (
    <Typography.Text className="px-2 py-1 text-xs rounded bg-gray-500/20 text-gray-400">
      {t(I18nKey.AZURE_DEVOPS$WEBHOOK_STATUS_NOT_INSTALLED)}
    </Typography.Text>
  );
}

export function AzureDevOpsWebhookManager({
  className,
}: AzureDevOpsWebhookManagerProps) {
  const { t } = useTranslation();
  const [isInstalling, setIsInstalling] = useState(false);
  const [isUninstalling, setIsUninstalling] = useState(false);

  const { data: status, isLoading, isError } = useAzureDevOpsResources(true);
  const reinstallMutation = useReinstallAzureDevOpsWebhook();
  const uninstallMutation = useUninstallAzureDevOpsWebhook();

  const handleReinstall = async () => {
    setIsInstalling(true);
    try {
      await reinstallMutation.mutateAsync();
    } finally {
      setIsInstalling(false);
    }
  };

  const handleUninstall = async () => {
    setIsUninstalling(true);
    try {
      await uninstallMutation.mutateAsync();
    } finally {
      setIsUninstalling(false);
    }
  };

  if (isLoading || isError || !status) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text
          className={cn("text-sm", isError ? "text-red-400" : "text-gray-400")}
        >
          {isError
            ? t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_ERROR)
            : t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_LOADING)}
        </Typography.Text>
      </div>
    );
  }

  const anyMutationPending = isInstalling || isUninstalling;
  const installDisabled = anyMutationPending || !status.webhook_secret_set;

  let installLabel: string;
  if (isInstalling) {
    installLabel = t(I18nKey.AZURE_DEVOPS$WEBHOOK_INSTALLING);
  } else if (status.webhook_installed) {
    installLabel = t(I18nKey.AZURE_DEVOPS$WEBHOOK_REINSTALL);
  } else {
    installLabel = t(I18nKey.AZURE_DEVOPS$WEBHOOK_INSTALL);
  }

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <Typography.H3 className="text-lg font-medium text-white">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_TITLE)}
      </Typography.H3>

      <Typography.Text className="text-sm text-gray-400">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_DESCRIPTION)}
      </Typography.Text>

      <div className="border border-neutral-700 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-neutral-800">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.AZURE_DEVOPS$WEBHOOK_COLUMN_ORGANIZATION)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.AZURE_DEVOPS$WEBHOOK_COLUMN_STATUS)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.AZURE_DEVOPS$WEBHOOK_COLUMN_ACTION)}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-700">
            <tr className="hover:bg-neutral-800/50 transition-colors align-top">
              <td className="px-4 py-3">
                <Typography.Text className="text-sm font-medium text-white">
                  {status.organization}
                </Typography.Text>
              </td>
              <td className="px-4 py-3">
                <StatusBadge status={status} />
              </td>
              <td className="px-4 py-3">
                <div className="flex gap-2">
                  <BrandButton
                    type="button"
                    variant="primary"
                    onClick={handleReinstall}
                    isDisabled={installDisabled}
                    className="cursor-pointer"
                    testId="azure-devops-install-webhook"
                  >
                    {installLabel}
                  </BrandButton>
                  {status.webhook_installed && (
                    <BrandButton
                      type="button"
                      variant="secondary"
                      onClick={handleUninstall}
                      isDisabled={anyMutationPending}
                      className="cursor-pointer"
                      testId="azure-devops-uninstall-webhook"
                    >
                      {isUninstalling
                        ? t(I18nKey.AZURE_DEVOPS$WEBHOOK_UNINSTALLING)
                        : t(I18nKey.AZURE_DEVOPS$WEBHOOK_UNINSTALL)}
                    </BrandButton>
                  )}
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
