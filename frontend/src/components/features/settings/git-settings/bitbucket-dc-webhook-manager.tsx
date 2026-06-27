import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { BitbucketDCResource } from "#/api/integration-service/integration-service.types";
import { useBitbucketDCResources } from "#/hooks/query/use-bitbucket-dc-resources-list";
import { useReinstallBitbucketDCWebhook } from "#/hooks/mutation/use-reinstall-bitbucket-dc-webhook";
import { useUninstallBitbucketDCWebhook } from "#/hooks/mutation/use-uninstall-bitbucket-dc-webhook";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface BitbucketDCWebhookManagerProps {
  className?: string;
}

function resourceKey(resource: BitbucketDCResource) {
  return `${resource.project_key}/${resource.repo_slug}`;
}

function StatusBadge({ enrolled }: { enrolled: boolean }) {
  const { t } = useTranslation();

  if (enrolled) {
    return (
      <Typography.Text className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
        {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_STATUS_ENROLLED)}
      </Typography.Text>
    );
  }

  return (
    <Typography.Text className="px-2 py-1 text-xs rounded bg-gray-500/20 text-gray-400">
      {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_STATUS_NOT_ENROLLED)}
    </Typography.Text>
  );
}

export function BitbucketDCWebhookManager({
  className,
}: BitbucketDCWebhookManagerProps) {
  const { t } = useTranslation();
  const [installingResource, setInstallingResource] = useState<string | null>(
    null,
  );
  const [uninstallingResource, setUninstallingResource] = useState<
    string | null
  >(null);

  const { data, isLoading, isError } = useBitbucketDCResources(true);
  const reinstallMutation = useReinstallBitbucketDCWebhook();
  const uninstallMutation = useUninstallBitbucketDCWebhook();

  const resources = data?.resources || [];

  const handleReinstall = async (resource: BitbucketDCResource) => {
    const key = resourceKey(resource);
    setInstallingResource(key);
    try {
      await reinstallMutation.mutateAsync({
        project_key: resource.project_key,
        repo_slug: resource.repo_slug,
      });
    } finally {
      setInstallingResource(null);
    }
  };

  const handleUninstall = async (resource: BitbucketDCResource) => {
    const key = resourceKey(resource);
    setUninstallingResource(key);
    try {
      await uninstallMutation.mutateAsync({
        project_key: resource.project_key,
        repo_slug: resource.repo_slug,
      });
    } finally {
      setUninstallingResource(null);
    }
  };

  if (isLoading) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text className="text-sm text-gray-400">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_LOADING)}
        </Typography.Text>
      </div>
    );
  }

  if (isError) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text className="text-sm text-red-400">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_ERROR)}
        </Typography.Text>
      </div>
    );
  }

  if (resources.length === 0) {
    return (
      <div className={cn("flex flex-col gap-4", className)}>
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
        <Typography.Text className="text-sm text-gray-400">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_NO_RESOURCES)}
        </Typography.Text>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <div className="flex items-center justify-between">
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
      </div>

      <Typography.Text className="text-sm text-gray-400">
        {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_MANAGER_DESCRIPTION)}
      </Typography.Text>

      <div className="border border-neutral-700 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-neutral-800">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_COLUMN_REPOSITORY)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_COLUMN_STATUS)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_COLUMN_ACTION)}
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

              let installLabel: string;
              if (isInstalling) {
                installLabel = t(
                  I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_INSTALLING,
                );
              } else if (resource.webhook_enrolled) {
                installLabel = t(
                  I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_REINSTALL,
                );
              } else {
                installLabel = t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_INSTALL);
              }

              return (
                <tr
                  key={key}
                  className="hover:bg-neutral-800/50 transition-colors align-top"
                >
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <Typography.Text className="text-sm font-medium text-white">
                        {resource.name}
                      </Typography.Text>
                      <Typography.Text className="text-xs text-gray-400">
                        {resource.full_name}
                      </Typography.Text>
                      {resource.installed_by_user_id && (
                        <Typography.Text className="text-xs text-gray-500">
                          {t(
                            I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_ENROLLED_BY,
                            {
                              userId: resource.installed_by_user_id,
                            },
                          )}
                        </Typography.Text>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge enrolled={resource.webhook_enrolled} />
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <BrandButton
                        type="button"
                        variant="primary"
                        onClick={() => handleReinstall(resource)}
                        isDisabled={anyMutationPending}
                        className="cursor-pointer"
                        testId={`bbdc-install-webhook-${key}`}
                      >
                        {installLabel}
                      </BrandButton>
                      {resource.webhook_enrolled && (
                        <BrandButton
                          type="button"
                          variant="secondary"
                          onClick={() => handleUninstall(resource)}
                          isDisabled={anyMutationPending}
                          className="cursor-pointer"
                          testId={`bbdc-uninstall-webhook-${key}`}
                        >
                          {isUninstalling
                            ? t(
                                I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_UNINSTALLING,
                              )
                            : t(
                                I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_UNINSTALL,
                              )}
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
