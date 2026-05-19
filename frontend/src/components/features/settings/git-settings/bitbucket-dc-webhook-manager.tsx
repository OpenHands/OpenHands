import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import type {
  BitbucketDCResource,
  BitbucketDCWebhookEnrollmentResult,
} from "#/api/integration-service/integration-service.types";
import { useBitbucketDCResources } from "#/hooks/query/use-bitbucket-dc-resources-list";
import { useEnrollBitbucketDCWebhook } from "#/hooks/mutation/use-enroll-bitbucket-dc-webhook";
import { useUpdateBitbucketDCWebhookId } from "#/hooks/mutation/use-update-bitbucket-dc-webhook-id";
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

function SetupValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <Typography.Text className="text-xs uppercase text-gray-500">
        {label}
      </Typography.Text>
      <CopyableContentWrapper text={value}>
        <code className="block rounded border border-neutral-700 bg-neutral-950 px-3 py-2 text-xs text-gray-200 break-all">
          {value}
        </code>
      </CopyableContentWrapper>
    </div>
  );
}

function EnrollmentSetup({
  result,
}: {
  result: BitbucketDCWebhookEnrollmentResult;
}) {
  const { t } = useTranslation();

  if (!result.success || !result.webhook_url || !result.webhook_secret) {
    return null;
  }

  return (
    <div className="mt-3 flex flex-col gap-3 rounded border border-neutral-700 bg-neutral-900/60 p-3">
      <Typography.Text className="text-sm text-gray-300">
        {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_SETUP_DESCRIPTION)}
      </Typography.Text>
      <div className="grid gap-3 md:grid-cols-2">
        <SetupValue
          label={t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_FIELD_NAME)}
          value={result.webhook_name}
        />
        <SetupValue
          label={t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_FIELD_URL)}
          value={result.webhook_url}
        />
        <SetupValue
          label={t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_FIELD_SECRET)}
          value={result.webhook_secret}
        />
        <SetupValue
          label={t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_FIELD_EVENTS)}
          value={result.events.join(", ")}
        />
      </div>
    </div>
  );
}

export function BitbucketDCWebhookManager({
  className,
}: BitbucketDCWebhookManagerProps) {
  const { t } = useTranslation();
  const [enrollingResource, setEnrollingResource] = useState<string | null>(
    null,
  );
  const [enrollmentResults, setEnrollmentResults] = useState<
    Map<string, BitbucketDCWebhookEnrollmentResult>
  >(new Map());
  const [webhookIds, setWebhookIds] = useState<Map<string, string>>(new Map());

  const { data, isLoading, isError } = useBitbucketDCResources(true);
  const enrollMutation = useEnrollBitbucketDCWebhook();
  const updateWebhookIdMutation = useUpdateBitbucketDCWebhookId();

  const resources = data?.resources || [];

  const handleEnroll = async (resource: BitbucketDCResource) => {
    const key = resourceKey(resource);
    setEnrollingResource(key);
    const nextResults = new Map(enrollmentResults);
    nextResults.delete(key);
    setEnrollmentResults(nextResults);

    try {
      const result = await enrollMutation.mutateAsync({
        project_key: resource.project_key,
        repo_slug: resource.repo_slug,
      });
      setEnrollmentResults(new Map(nextResults).set(key, result));
    } finally {
      setEnrollingResource(null);
    }
  };

  const handleWebhookIdChange = (key: string, value: string) => {
    const nextWebhookIds = new Map(webhookIds);
    nextWebhookIds.set(key, value);
    setWebhookIds(nextWebhookIds);
  };

  const handleWebhookIdSave = async (resource: BitbucketDCResource) => {
    const key = resourceKey(resource);
    const webhookId = (webhookIds.get(key) ?? resource.webhook_id ?? "").trim();
    if (!webhookId) {
      return;
    }

    await updateWebhookIdMutation.mutateAsync({
      resource: {
        project_key: resource.project_key,
        repo_slug: resource.repo_slug,
      },
      webhookId,
    });
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
                {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_COLUMN_WEBHOOK_ID)}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_COLUMN_ACTION)}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-700">
            {resources.map((resource) => {
              const key = resourceKey(resource);
              const result = enrollmentResults.get(key);
              const isEnrolling = enrollingResource === key;
              const webhookId =
                webhookIds.get(key) ?? resource.webhook_id ?? "";
              let enrollButtonLabel = t(
                I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_ENROLL,
              );
              if (resource.webhook_enrolled) {
                enrollButtonLabel = t(
                  I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_REGENERATE,
                );
              }
              if (isEnrolling) {
                enrollButtonLabel = t(
                  I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_GENERATING,
                );
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
                      {result && <EnrollmentSetup result={result} />}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge enrolled={resource.webhook_enrolled} />
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={webhookId}
                        onChange={(event) =>
                          handleWebhookIdChange(key, event.target.value)
                        }
                        placeholder={t(
                          I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_OPTIONAL_PLACEHOLDER,
                        )}
                        className="w-28 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-white"
                        data-testid={`bbdc-webhook-id-${key}`}
                      />
                      <BrandButton
                        type="button"
                        variant="secondary"
                        onClick={() => handleWebhookIdSave(resource)}
                        isDisabled={
                          updateWebhookIdMutation.isPending || !webhookId.trim()
                        }
                        testId={`bbdc-save-webhook-id-${key}`}
                      >
                        {t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_SAVE)}
                      </BrandButton>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <BrandButton
                      type="button"
                      variant="primary"
                      onClick={() => handleEnroll(resource)}
                      isDisabled={enrollingResource !== null}
                      className="cursor-pointer"
                      testId={`bbdc-enroll-webhook-${key}`}
                    >
                      {enrollButtonLabel}
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
