import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useGitLabResources } from "#/hooks/query/use-gitlab-resources-list";
import { useReinstallGitLabWebhook } from "#/hooks/mutation/use-reinstall-gitlab-webhook";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { GitLabResource } from "#/api/integration-service/integration-service.types";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { WebhookStatusBadge } from "./webhook-status-badge";
import { GitLabWebhookManagerState } from "./gitlab-webhook-manager-state";

interface GitLabWebhookManagerProps {
  className?: string;
}

export function GitLabWebhookManager({ className }: GitLabWebhookManagerProps) {
  const { t } = useTranslation();
  const [installingResource, setInstallingResource] = useState<string | null>(
    null,
  );
  const [installationResults, setInstallationResults] = useState<
    Map<string, { success: boolean; error: string | null }>
  >(new Map());

  const { data, isLoading, isError } = useGitLabResources(true);
  const reinstallMutation = useReinstallGitLabWebhook();

  const resources = data?.resources || [];

  const handleReinstall = async (resource: GitLabResource) => {
    const key = `${resource.type}:${resource.id}`;
    setInstallingResource(key);

    // Clear previous result for this resource
    const newResults = new Map(installationResults);
    newResults.delete(key);
    setInstallationResults(newResults);

    try {
      const result = await reinstallMutation.mutateAsync({
        type: resource.type,
        id: resource.id,
      });

      // Store result for display
      const resultsMap = new Map(installationResults);
      resultsMap.set(key, {
        success: result.success,
        error: result.error,
      });
      setInstallationResults(resultsMap);
    } catch (error: unknown) {
      // Store error result
      const resultsMap = new Map(installationResults);
      const errorMessage =
        error instanceof Error
          ? error.message
          : t(I18nKey.GITLAB$WEBHOOK_REINSTALL_FAILED);
      resultsMap.set(key, {
        success: false,
        error: errorMessage,
      });
      setInstallationResults(resultsMap);
    } finally {
      setInstallingResource(null);
    }
  };

  const getResourceKey = (resource: GitLabResource) =>
    `${resource.type}:${resource.id}`;

  if (isLoading) {
    return (
      <GitLabWebhookManagerState
        className={className}
        titleKey={I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE}
        messageKey={I18nKey.GITLAB$WEBHOOK_MANAGER_LOADING}
      />
    );
  }

  if (isError) {
    return (
      <GitLabWebhookManagerState
        className={className}
        titleKey={I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE}
        messageKey={I18nKey.GITLAB$WEBHOOK_MANAGER_ERROR}
        messageColor="text-red-400"
      />
    );
  }

  if (resources.length === 0) {
    return (
      <GitLabWebhookManagerState
        className={className}
        titleKey={I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE}
        messageKey={I18nKey.GITLAB$WEBHOOK_MANAGER_NO_RESOURCES}
      />
    );
  }

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <div className="flex items-center justify-between">
        <Typography.H3 className="text-lg font-medium text-white">
          {t(I18nKey.GITLAB$WEBHOOK_MANAGER_TITLE)}
        </Typography.H3>
      </div>

      <Typography.Text className="text-sm text-gray-400">
        {t(I18nKey.GITLAB$WEBHOOK_MANAGER_DESCRIPTION)}
      </Typography.Text>

      <div className="border border-neutral-700 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-neutral-800">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.GITLAB$WEBHOOK_COLUMN_RESOURCE)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.GITLAB$WEBHOOK_COLUMN_TYPE)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.GITLAB$WEBHOOK_COLUMN_STATUS)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.GITLAB$WEBHOOK_COLUMN_ACTION)}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-700">
            {resources.map((resource) => {
              const key = getResourceKey(resource);
              const result = installationResults.get(key);
              const isInstalling = installingResource === key;

              return (
                <tr
                  key={key}
                  className="hover:bg-neutral-800/50 transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <Typography.Text className="text-sm font-medium text-white">
                        {resource.name}
                      </Typography.Text>
                      <Typography.Text className="text-xs text-gray-400">
                        {resource.full_path}
                      </Typography.Text>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Typography.Text className="text-sm text-gray-300 capitalize">
                      {resource.type}
                    </Typography.Text>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-col gap-1">
                      <WebhookStatusBadge
                        webhookInstalled={resource.webhook_installed}
                        installationResult={result}
                      />
                      {result?.error && (
                        <Typography.Text className="text-xs text-red-400">
                          {result.error}
                        </Typography.Text>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <BrandButton
                      type="button"
                      variant="primary"
                      onClick={() => handleReinstall(resource)}
                      isDisabled={
                        installingResource !== null ||
                        resource.webhook_installed ||
                        result?.success === true
                      }
                      className="cursor-pointer"
                      testId={`reinstall-webhook-button-${key}`}
                    >
                      {isInstalling
                        ? t(I18nKey.GITLAB$WEBHOOK_REINSTALLING)
                        : t(I18nKey.GITLAB$WEBHOOK_REINSTALL)}
                    </BrandButton>
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
