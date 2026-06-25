import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { AzureDevOpsResource } from "#/api/integration-service/integration-service.types";
import { useAzureDevOpsResources } from "#/hooks/query/use-azure-devops-resources-list";
import { useReinstallAzureDevOpsWebhook } from "#/hooks/mutation/use-reinstall-azure-devops-webhook";
import { useUninstallAzureDevOpsWebhook } from "#/hooks/mutation/use-uninstall-azure-devops-webhook";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface AzureDevOpsWebhookManagerProps {
  className?: string;
}

function resourceKey(resource: AzureDevOpsResource) {
  return resource.project_id;
}

function resourceIdentifier(resource: AzureDevOpsResource) {
  return {
    organization: resource.organization,
    project_id: resource.project_id,
    project_name: resource.project_name,
  };
}

function StatusBadge({ resource }: { resource: AzureDevOpsResource }) {
  const { t } = useTranslation();

  if (!resource.webhook_secret_set) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-red-500/20 text-red-400">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_STATUS_MISSING_SECRET)}
      </Typography.Text>
    );
  }

  if (resource.webhook_installed) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_STATUS_INSTALLED)}
      </Typography.Text>
    );
  }

  if (resource.pr_webhook_installed || resource.work_item_webhook_installed) {
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
  const [installingResource, setInstallingResource] = useState<string | null>(
    null,
  );
  const [uninstallingResource, setUninstallingResource] = useState<
    string | null
  >(null);

  const { data, isLoading, isError } = useAzureDevOpsResources(true);
  const reinstallMutation = useReinstallAzureDevOpsWebhook();
  const uninstallMutation = useUninstallAzureDevOpsWebhook();

  const resources = data?.resources || [];

  const handleReinstall = async (resource: AzureDevOpsResource) => {
    const key = resourceKey(resource);
    setInstallingResource(key);
    try {
      await reinstallMutation.mutateAsync(resourceIdentifier(resource));
    } finally {
      setInstallingResource(null);
    }
  };

  const handleUninstall = async (resource: AzureDevOpsResource) => {
    const key = resourceKey(resource);
    setUninstallingResource(key);
    try {
      await uninstallMutation.mutateAsync(resourceIdentifier(resource));
    } finally {
      setUninstallingResource(null);
    }
  };

  if (isLoading) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text className="text-sm text-gray-400">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_LOADING)}
        </Typography.Text>
      </div>
    );
  }

  if (isError) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text className="text-sm text-red-400">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_ERROR)}
        </Typography.Text>
      </div>
    );
  }

  if (resources.length === 0) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text className="text-sm text-gray-400">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_NO_RESOURCES)}
        </Typography.Text>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <div className="flex items-center justify-between">
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
      </div>

      <Typography.Text className="text-sm text-gray-400">
        {t(I18nKey.AZURE_DEVOPS$WEBHOOK_MANAGER_DESCRIPTION)}
      </Typography.Text>

      <div className="border border-neutral-700 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-neutral-800">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.AZURE_DEVOPS$WEBHOOK_COLUMN_PROJECT)}
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
            {resources.map((resource) => {
              const key = resourceKey(resource);
              const isInstalling = installingResource === key;
              const isUninstalling = uninstallingResource === key;
              const anyMutationPending =
                installingResource !== null || uninstallingResource !== null;
              const installDisabled =
                anyMutationPending || !resource.webhook_secret_set;

              let installLabel: string;
              if (isInstalling) {
                installLabel = t(I18nKey.AZURE_DEVOPS$WEBHOOK_INSTALLING);
              } else if (resource.webhook_installed) {
                installLabel = t(I18nKey.AZURE_DEVOPS$WEBHOOK_REINSTALL);
              } else {
                installLabel = t(I18nKey.AZURE_DEVOPS$WEBHOOK_INSTALL);
              }

              return (
                <tr
                  key={key}
                  className="hover:bg-neutral-800/50 transition-colors align-top"
                >
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <Typography.Text className="text-sm font-medium text-white">
                        {resource.project_name}
                      </Typography.Text>
                      <Typography.Text className="text-xs text-gray-400">
                        {resource.full_name}
                      </Typography.Text>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge resource={resource} />
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <BrandButton
                        type="button"
                        variant="primary"
                        onClick={() => handleReinstall(resource)}
                        isDisabled={installDisabled}
                        className="cursor-pointer"
                        testId={`azure-devops-install-webhook-${key}`}
                      >
                        {installLabel}
                      </BrandButton>
                      {resource.webhook_installed && (
                        <BrandButton
                          type="button"
                          variant="secondary"
                          onClick={() => handleUninstall(resource)}
                          isDisabled={anyMutationPending}
                          className="cursor-pointer"
                          testId={`azure-devops-uninstall-webhook-${key}`}
                        >
                          {isUninstalling
                            ? t(I18nKey.AZURE_DEVOPS$WEBHOOK_UNINSTALLING)
                            : t(I18nKey.AZURE_DEVOPS$WEBHOOK_UNINSTALL)}
                        </BrandButton>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
