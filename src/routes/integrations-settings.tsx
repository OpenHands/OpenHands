import React from "react";
import { Navigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import { useAppwriteIntegration } from "#/hooks/query/use-appwrite-integration";
import { useDependencyTrackIntegration } from "#/hooks/query/use-dependency-track-integration";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useSettings } from "#/hooks/query/use-settings";
import { AppwriteService } from "#/api/integrations/appwrite-service";
import { DependencyTrackService } from "#/api/integrations/dependency-track-service";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { DEFAULT_APPWRITE_ENDPOINT } from "#/utils/appwrite-integration-secrets";
import { buildAppwriteIntegrationsPatch } from "#/utils/appwrite-workspace-config";
import { buildDependencyTrackIntegrationsPatch } from "#/utils/dependency-track-workspace-config";
import { cn } from "#/utils/utils";

export const handle = { hideTitle: true };

export function IntegrationsSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const { data: workspacesData, isLoading: workspacesLoading } =
    useLocalWorkspaces();
  const workspaces = workspacesData?.workspaces ?? [];
  const [selectedWorkspaceId, setSelectedWorkspaceId] = React.useState<
    string | null
  >(null);

  React.useEffect(() => {
    if (!selectedWorkspaceId && workspaces.length > 0) {
      setSelectedWorkspaceId(workspaces[0].id);
    }
  }, [workspaces, selectedWorkspaceId]);

  const { config, apiKeyIsSet, secretName, isLoading } =
    useAppwriteIntegration(selectedWorkspaceId);
  const {
    config: dtConfig,
    apiKeyIsSet: dtApiKeyIsSet,
    secretName: dtSecretName,
    isLoading: dtConfigLoading,
  } = useDependencyTrackIntegration(selectedWorkspaceId);
  const { mutateAsync: saveSettings, isPending: isSaving } = useSaveSettings();
  const { mutateAsync: createSecret, isPending: isSavingSecret } =
    useCreateSecret();

  const [enabled, setEnabled] = React.useState(false);
  const [endpoint, setEndpoint] = React.useState(DEFAULT_APPWRITE_ENDPOINT);
  const [projectId, setProjectId] = React.useState("");
  const [apiKey, setApiKey] = React.useState("");
  const [isTesting, setIsTesting] = React.useState(false);

  const [dtEnabled, setDtEnabled] = React.useState(false);
  const [dtBaseUrl, setDtBaseUrl] = React.useState("");
  const [dtProjectUuid, setDtProjectUuid] = React.useState("");
  const [dtApiKey, setDtApiKey] = React.useState("");
  const [isDtTesting, setIsDtTesting] = React.useState(false);

  React.useEffect(() => {
    setEnabled(config.enabled);
    setEndpoint(config.endpoint || DEFAULT_APPWRITE_ENDPOINT);
    setProjectId(config.projectId);
    setApiKey("");
  }, [
    selectedWorkspaceId,
    config.enabled,
    config.endpoint,
    config.projectId,
  ]);

  React.useEffect(() => {
    setDtEnabled(dtConfig.enabled);
    setDtBaseUrl(dtConfig.baseUrl);
    setDtProjectUuid(dtConfig.projectUuid);
    setDtApiKey("");
  }, [
    selectedWorkspaceId,
    dtConfig.enabled,
    dtConfig.baseUrl,
    dtConfig.projectUuid,
  ]);

  if (backend.kind !== "local") {
    return <Navigate to="/settings/agents" replace />;
  }

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
    queryClient.invalidateQueries({ queryKey: ["secrets-search"] });
    queryClient.invalidateQueries({ queryKey: ["secrets"] });
  };

  const workspaceOptions = workspaces.map((workspace) => ({
    key: workspace.id,
    label: workspace.name,
  }));

  const handleSave = async () => {
    if (!selectedWorkspaceId || !secretName) {
      return;
    }
    try {
      if (apiKey.trim()) {
        await createSecret({
          name: secretName,
          value: apiKey.trim(),
          description: `AppWrite API key for workspace ${selectedWorkspaceId}`,
        });
        setApiKey("");
      }
      await saveSettings({
        integrations: buildAppwriteIntegrationsPatch(
          settings?.integrations,
          selectedWorkspaceId,
          {
            enabled,
            endpoint: endpoint.trim(),
            projectId: projectId.trim(),
            apiKeySecretName: secretName,
          },
        ),
      });
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const handleTest = async () => {
    if (!selectedWorkspaceId) {
      return;
    }
    setIsTesting(true);
    try {
      await AppwriteService.forWorkspace(selectedWorkspaceId).testConnection();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$TEST_SUCCESS));
    } catch (error) {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) ||
          t(I18nKey.INTEGRATIONS$TEST_FAILED),
      );
    } finally {
      setIsTesting(false);
    }
  };

  const handleDtSave = async () => {
    if (!selectedWorkspaceId || !dtSecretName) {
      return;
    }
    try {
      if (dtApiKey.trim()) {
        await createSecret({
          name: dtSecretName,
          value: dtApiKey.trim(),
          description: `Dependency-Track API key for workspace ${selectedWorkspaceId}`,
        });
        setDtApiKey("");
      }
      await saveSettings({
        integrations: {
          ...settings?.integrations,
          ...buildDependencyTrackIntegrationsPatch(
            settings?.integrations,
            selectedWorkspaceId,
            {
              enabled: dtEnabled,
              baseUrl: dtBaseUrl.trim(),
              projectUuid: dtProjectUuid.trim(),
              apiKeySecretName: dtSecretName,
            },
          ),
        },
      });
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const handleDtTest = async () => {
    if (!selectedWorkspaceId) {
      return;
    }
    setIsDtTesting(true);
    try {
      await DependencyTrackService.forWorkspace(
        selectedWorkspaceId,
      ).testConnection();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$TEST_SUCCESS));
    } catch (error) {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) ||
          t(I18nKey.INTEGRATIONS$TEST_FAILED),
      );
    } finally {
      setIsDtTesting(false);
    }
  };

  if (isLoading || workspacesLoading || dtConfigLoading) {
    return (
      <div className="p-4" data-testid="integrations-settings-loading">
        <Typography.Text>{t(I18nKey.HOME$LOADING)}</Typography.Text>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col gap-6 p-4 md:p-0"
      data-testid="integrations-settings"
    >
      <div>
        <Typography.H2 className="text-lg font-semibold text-white">
          {t(I18nKey.INTEGRATIONS$TITLE)}
        </Typography.H2>
        <Typography.Text className="mt-1 text-sm text-[var(--oh-muted)]">
          {t(I18nKey.SETTINGS$PAGE_INTEGRATIONS_SUBLINE)}
        </Typography.Text>
      </div>

      <section
        className={cn(
          "rounded-lg border border-[var(--oh-border)]",
          "bg-[var(--oh-surface)] p-4 flex flex-col gap-4",
        )}
        data-testid="appwrite-integration-card"
      >
        <div>
          <Typography.Text className="text-base font-medium text-white">
            {t(I18nKey.INTEGRATIONS$APPWRITE_NAME)}
          </Typography.Text>
          <Typography.Text className="mt-1 block text-sm text-[var(--oh-muted)]">
            {t(I18nKey.INTEGRATIONS$APPWRITE_DESCRIPTION)}
          </Typography.Text>
        </div>

        {workspaceOptions.length === 0 ? (
          <Typography.Text
            className="text-sm text-[var(--oh-muted)]"
            testId="appwrite-no-workspaces"
          >
            {t(I18nKey.INTEGRATIONS$NO_WORKSPACES)}
          </Typography.Text>
        ) : (
          <>
            <SettingsDropdownInput
              testId="appwrite-workspace"
              name="appwrite-workspace"
              label={t(I18nKey.INTEGRATIONS$WORKSPACE)}
              items={workspaceOptions}
              selectedKey={selectedWorkspaceId ?? undefined}
              onSelectionChange={(key) => {
                if (typeof key === "string") {
                  setSelectedWorkspaceId(key);
                }
              }}
            />

            <label className="flex items-center gap-2 text-sm text-white">
              <input
                type="checkbox"
                data-testid="appwrite-enabled"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
                className="size-4"
              />
              {t(I18nKey.INTEGRATIONS$ENABLED)}
            </label>

            <SettingsInput
              testId="appwrite-endpoint"
              label={t(I18nKey.INTEGRATIONS$ENDPOINT)}
              type="url"
              value={endpoint}
              onChange={setEndpoint}
              placeholder={DEFAULT_APPWRITE_ENDPOINT}
            />

            <SettingsInput
              testId="appwrite-project-id"
              label={t(I18nKey.INTEGRATIONS$PROJECT_ID)}
              type="text"
              value={projectId}
              onChange={setProjectId}
            />

            <div className="flex flex-col gap-1">
              <SettingsInput
                testId="appwrite-api-key"
                label={t(I18nKey.INTEGRATIONS$API_KEY)}
                type="password"
                value={apiKey}
                onChange={setApiKey}
                placeholder={t(I18nKey.INTEGRATIONS$API_KEY_PLACEHOLDER)}
              />
              {apiKeyIsSet && (
                <Typography.Text
                  className="text-xs text-[var(--oh-muted)]"
                  testId="appwrite-api-key-set"
                >
                  {t(I18nKey.INTEGRATIONS$API_KEY_SET)}
                </Typography.Text>
              )}
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              <BrandButton
                type="button"
                variant="primary"
                testId="appwrite-save"
                onClick={() => void handleSave()}
                isDisabled={
                  isSaving || isSavingSecret || !selectedWorkspaceId
                }
              >
                {t(I18nKey.INTEGRATIONS$SAVE)}
              </BrandButton>
              <BrandButton
                type="button"
                variant="secondary"
                testId="appwrite-test"
                onClick={() => void handleTest()}
                isDisabled={isTesting || !enabled || !selectedWorkspaceId}
              >
                {t(I18nKey.INTEGRATIONS$TEST_CONNECTION)}
              </BrandButton>
            </div>
          </>
        )}
      </section>

      <section
        className={cn(
          "rounded-lg border border-[var(--oh-border)]",
          "bg-[var(--oh-surface)] p-4 flex flex-col gap-4",
        )}
        data-testid="dependency-track-integration-card"
      >
        <div>
          <Typography.Text className="text-base font-medium text-white">
            {t(I18nKey.INTEGRATIONS$DEPENDENCY_TRACK_NAME)}
          </Typography.Text>
          <Typography.Text className="mt-1 block text-sm text-[var(--oh-muted)]">
            {t(I18nKey.INTEGRATIONS$DEPENDENCY_TRACK_DESCRIPTION)}
          </Typography.Text>
        </div>

        {workspaceOptions.length === 0 ? (
          <Typography.Text
            className="text-sm text-[var(--oh-muted)]"
            testId="dependency-track-no-workspaces"
          >
            {t(I18nKey.INTEGRATIONS$NO_WORKSPACES)}
          </Typography.Text>
        ) : (
          <>
            <SettingsDropdownInput
              testId="dependency-track-workspace"
              name="dependency-track-workspace"
              label={t(I18nKey.INTEGRATIONS$WORKSPACE)}
              items={workspaceOptions}
              selectedKey={selectedWorkspaceId ?? undefined}
              onSelectionChange={(key) => {
                if (typeof key === "string") {
                  setSelectedWorkspaceId(key);
                }
              }}
            />

            <label className="flex items-center gap-2 text-sm text-white">
              <input
                type="checkbox"
                data-testid="dependency-track-enabled"
                checked={dtEnabled}
                onChange={(e) => setDtEnabled(e.target.checked)}
                className="size-4"
              />
              {t(I18nKey.INTEGRATIONS$ENABLED)}
            </label>

            <SettingsInput
              testId="dependency-track-base-url"
              label={t(I18nKey.INTEGRATIONS$BASE_URL)}
              type="url"
              value={dtBaseUrl}
              onChange={setDtBaseUrl}
              placeholder={t(I18nKey.INTEGRATIONS$BASE_URL_PLACEHOLDER)}
            />

            <SettingsInput
              testId="dependency-track-project-uuid"
              label={t(I18nKey.INTEGRATIONS$PROJECT_UUID)}
              type="text"
              value={dtProjectUuid}
              onChange={setDtProjectUuid}
            />

            <div className="flex flex-col gap-1">
              <SettingsInput
                testId="dependency-track-api-key"
                label={t(I18nKey.INTEGRATIONS$API_KEY)}
                type="password"
                value={dtApiKey}
                onChange={setDtApiKey}
                placeholder={t(I18nKey.INTEGRATIONS$API_KEY_PLACEHOLDER)}
              />
              {dtApiKeyIsSet && (
                <Typography.Text
                  className="text-xs text-[var(--oh-muted)]"
                  testId="dependency-track-api-key-set"
                >
                  {t(I18nKey.INTEGRATIONS$API_KEY_SET)}
                </Typography.Text>
              )}
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              <BrandButton
                type="button"
                variant="primary"
                testId="dependency-track-save"
                onClick={() => void handleDtSave()}
                isDisabled={
                  isSaving || isSavingSecret || !selectedWorkspaceId
                }
              >
                {t(I18nKey.INTEGRATIONS$SAVE)}
              </BrandButton>
              <BrandButton
                type="button"
                variant="secondary"
                testId="dependency-track-test"
                onClick={() => void handleDtTest()}
                isDisabled={isDtTesting || !dtEnabled || !selectedWorkspaceId}
              >
                {t(I18nKey.INTEGRATIONS$TEST_CONNECTION)}
              </BrandButton>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

export default IntegrationsSettingsScreen;
