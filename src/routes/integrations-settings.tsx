import React from "react";
import { Navigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import { useAppwriteIntegration } from "#/hooks/query/use-appwrite-integration";
import { useDependencyTrackIntegration } from "#/hooks/query/use-dependency-track-integration";
import { usePlaneIntegration } from "#/hooks/query/use-plane-integration";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useSettings } from "#/hooks/query/use-settings";
import { AppwriteService } from "#/api/integrations/appwrite-service";
import { DependencyTrackService } from "#/api/integrations/dependency-track-service";
import { PlaneService } from "#/api/integrations/plane-service";
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
import { buildPlaneIntegrationsPatch } from "#/utils/plane-workspace-config";
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

  const appwrite = useAppwriteIntegration(selectedWorkspaceId);
  const plane = usePlaneIntegration(selectedWorkspaceId);
  const dependencyTrack = useDependencyTrackIntegration(selectedWorkspaceId);
  const { mutateAsync: saveSettings, isPending: isSaving } = useSaveSettings();
  const { mutateAsync: createSecret, isPending: isSavingSecret } =
    useCreateSecret();

  const [appwriteEnabled, setAppwriteEnabled] = React.useState(false);
  const [appwriteEndpoint, setAppwriteEndpoint] = React.useState(
    DEFAULT_APPWRITE_ENDPOINT,
  );
  const [appwriteProjectId, setAppwriteProjectId] = React.useState("");
  const [appwriteApiKey, setAppwriteApiKey] = React.useState("");
  const [isTestingAppwrite, setIsTestingAppwrite] = React.useState(false);

  const [planeEnabled, setPlaneEnabled] = React.useState(false);
  const [planeBaseUrl, setPlaneBaseUrl] = React.useState("");
  const [planeWorkspaceSlug, setPlaneWorkspaceSlug] = React.useState("");
  const [planeProjectId, setPlaneProjectId] = React.useState("");
  const [planeModuleId, setPlaneModuleId] = React.useState("");
  const [planeApiKey, setPlaneApiKey] = React.useState("");
  const [isTestingPlane, setIsTestingPlane] = React.useState(false);

  const [dtEnabled, setDtEnabled] = React.useState(false);
  const [dtBaseUrl, setDtBaseUrl] = React.useState("");
  const [dtProjectUuid, setDtProjectUuid] = React.useState("");
  const [dtApiKey, setDtApiKey] = React.useState("");
  const [isDtTesting, setIsDtTesting] = React.useState(false);

  React.useEffect(() => {
    setAppwriteEnabled(appwrite.config.enabled);
    setAppwriteEndpoint(appwrite.config.endpoint || DEFAULT_APPWRITE_ENDPOINT);
    setAppwriteProjectId(appwrite.config.projectId);
    setAppwriteApiKey("");
  }, [
    selectedWorkspaceId,
    appwrite.config.enabled,
    appwrite.config.endpoint,
    appwrite.config.projectId,
  ]);

  React.useEffect(() => {
    setPlaneEnabled(plane.config.enabled);
    setPlaneBaseUrl(plane.config.baseUrl);
    setPlaneWorkspaceSlug(plane.config.workspaceSlug);
    setPlaneProjectId(plane.config.projectId);
    setPlaneModuleId(plane.config.moduleId ?? "");
    setPlaneApiKey("");
  }, [
    selectedWorkspaceId,
    plane.config.enabled,
    plane.config.baseUrl,
    plane.config.workspaceSlug,
    plane.config.projectId,
    plane.config.moduleId,
  ]);

  React.useEffect(() => {
    setDtEnabled(dependencyTrack.config.enabled);
    setDtBaseUrl(dependencyTrack.config.baseUrl);
    setDtProjectUuid(dependencyTrack.config.projectUuid);
    setDtApiKey("");
  }, [
    selectedWorkspaceId,
    dependencyTrack.config.enabled,
    dependencyTrack.config.baseUrl,
    dependencyTrack.config.projectUuid,
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

  const handleSaveAppwrite = async () => {
    if (!selectedWorkspaceId || !appwrite.secretName) {
      return;
    }
    try {
      if (appwriteApiKey.trim()) {
        await createSecret({
          name: appwrite.secretName,
          value: appwriteApiKey.trim(),
          description: `AppWrite API key for workspace ${selectedWorkspaceId}`,
        });
        setAppwriteApiKey("");
      }
      await saveSettings({
        integrations: buildAppwriteIntegrationsPatch(
          settings?.integrations,
          selectedWorkspaceId,
          {
            enabled: appwriteEnabled,
            endpoint: appwriteEndpoint.trim(),
            projectId: appwriteProjectId.trim(),
            apiKeySecretName: appwrite.secretName,
          },
        ),
      });
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const handleTestAppwrite = async () => {
    if (!selectedWorkspaceId) {
      return;
    }
    setIsTestingAppwrite(true);
    try {
      await AppwriteService.forWorkspace(selectedWorkspaceId).testConnection();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$TEST_SUCCESS));
    } catch (error) {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) ||
          t(I18nKey.INTEGRATIONS$TEST_FAILED),
      );
    } finally {
      setIsTestingAppwrite(false);
    }
  };

  const handleSavePlane = async () => {
    if (!selectedWorkspaceId || !plane.secretName) {
      return;
    }
    try {
      if (planeApiKey.trim()) {
        await createSecret({
          name: plane.secretName,
          value: planeApiKey.trim(),
          description: `Plane API key for workspace ${selectedWorkspaceId}`,
        });
        setPlaneApiKey("");
      }
      await saveSettings({
        integrations: buildPlaneIntegrationsPatch(
          settings?.integrations,
          selectedWorkspaceId,
          {
            enabled: planeEnabled,
            baseUrl: planeBaseUrl.trim(),
            workspaceSlug: planeWorkspaceSlug.trim(),
            projectId: planeProjectId.trim(),
            moduleId: planeModuleId.trim(),
            apiKeySecretName: plane.secretName,
          },
        ),
      });
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const handleTestPlane = async () => {
    if (!selectedWorkspaceId) {
      return;
    }
    setIsTestingPlane(true);
    try {
      await PlaneService.forWorkspace(selectedWorkspaceId).testConnection();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$TEST_SUCCESS));
    } catch (error) {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) ||
          t(I18nKey.INTEGRATIONS$TEST_FAILED),
      );
    } finally {
      setIsTestingPlane(false);
    }
  };

  const handleSaveDependencyTrack = async () => {
    if (!selectedWorkspaceId || !dependencyTrack.secretName) {
      return;
    }
    try {
      if (dtApiKey.trim()) {
        await createSecret({
          name: dependencyTrack.secretName,
          value: dtApiKey.trim(),
          description: `Dependency-Track API key for workspace ${selectedWorkspaceId}`,
        });
        setDtApiKey("");
      }
      await saveSettings({
        integrations: buildDependencyTrackIntegrationsPatch(
          settings?.integrations,
          selectedWorkspaceId,
          {
            enabled: dtEnabled,
            baseUrl: dtBaseUrl.trim(),
            projectUuid: dtProjectUuid.trim(),
            apiKeySecretName: dependencyTrack.secretName,
          },
        ),
      });
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const handleTestDependencyTrack = async () => {
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

  if (
    appwrite.isLoading ||
    plane.isLoading ||
    dependencyTrack.isLoading ||
    workspacesLoading
  ) {
    return (
      <div className="p-4" data-testid="integrations-settings-loading">
        <Typography.Text>{t(I18nKey.HOME$LOADING)}</Typography.Text>
      </div>
    );
  }

  const workspaceSelector = (
    <SettingsDropdownInput
      testId="integrations-workspace"
      name="integrations-workspace"
      label={t(I18nKey.INTEGRATIONS$WORKSPACE)}
      items={workspaceOptions}
      selectedKey={selectedWorkspaceId ?? undefined}
      onSelectionChange={(key) => {
        if (typeof key === "string") {
          setSelectedWorkspaceId(key);
        }
      }}
    />
  );

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

      {workspaceOptions.length === 0 ? (
        <Typography.Text
          className="text-sm text-[var(--oh-muted)]"
          testId="appwrite-no-workspaces"
        >
          {t(I18nKey.INTEGRATIONS$NO_WORKSPACES)}
        </Typography.Text>
      ) : (
        <>
          {workspaceSelector}

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

            <label className="flex items-center gap-2 text-sm text-white">
              <input
                type="checkbox"
                data-testid="appwrite-enabled"
                checked={appwriteEnabled}
                onChange={(e) => setAppwriteEnabled(e.target.checked)}
                className="size-4"
              />
              {t(I18nKey.INTEGRATIONS$ENABLED)}
            </label>

            <SettingsInput
              testId="appwrite-endpoint"
              label={t(I18nKey.INTEGRATIONS$ENDPOINT)}
              type="url"
              value={appwriteEndpoint}
              onChange={setAppwriteEndpoint}
              placeholder={DEFAULT_APPWRITE_ENDPOINT}
            />

            <SettingsInput
              testId="appwrite-project-id"
              label={t(I18nKey.INTEGRATIONS$PROJECT_ID)}
              type="text"
              value={appwriteProjectId}
              onChange={setAppwriteProjectId}
            />

            <div className="flex flex-col gap-1">
              <SettingsInput
                testId="appwrite-api-key"
                label={t(I18nKey.INTEGRATIONS$API_KEY)}
                type="password"
                value={appwriteApiKey}
                onChange={setAppwriteApiKey}
                placeholder={t(I18nKey.INTEGRATIONS$API_KEY_PLACEHOLDER)}
              />
              {appwrite.apiKeyIsSet && (
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
                onClick={() => void handleSaveAppwrite()}
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
                onClick={() => void handleTestAppwrite()}
                isDisabled={
                  isTestingAppwrite || !appwriteEnabled || !selectedWorkspaceId
                }
              >
                {t(I18nKey.INTEGRATIONS$TEST_CONNECTION)}
              </BrandButton>
            </div>
          </section>

          <section
            className={cn(
              "rounded-lg border border-[var(--oh-border)]",
              "bg-[var(--oh-surface)] p-4 flex flex-col gap-4",
            )}
            data-testid="plane-integration-card"
          >
            <div>
              <Typography.Text className="text-base font-medium text-white">
                {t(I18nKey.INTEGRATIONS$PLANE_NAME)}
              </Typography.Text>
              <Typography.Text className="mt-1 block text-sm text-[var(--oh-muted)]">
                {t(I18nKey.INTEGRATIONS$PLANE_DESCRIPTION)}
              </Typography.Text>
            </div>

            <label className="flex items-center gap-2 text-sm text-white">
              <input
                type="checkbox"
                data-testid="plane-enabled"
                checked={planeEnabled}
                onChange={(e) => setPlaneEnabled(e.target.checked)}
                className="size-4"
              />
              {t(I18nKey.INTEGRATIONS$ENABLED)}
            </label>

            <SettingsInput
              testId="plane-base-url"
              label={t(I18nKey.INTEGRATIONS$PLANE_URL)}
              type="url"
              value={planeBaseUrl}
              onChange={setPlaneBaseUrl}
              placeholder={t(I18nKey.INTEGRATIONS$PLANE_URL_PLACEHOLDER)}
            />

            <SettingsInput
              testId="plane-workspace-slug"
              label={t(I18nKey.INTEGRATIONS$PLANE_WORKSPACE_SLUG)}
              type="text"
              value={planeWorkspaceSlug}
              onChange={setPlaneWorkspaceSlug}
              placeholder={t(
                I18nKey.INTEGRATIONS$PLANE_WORKSPACE_SLUG_PLACEHOLDER,
              )}
            />

            <SettingsInput
              testId="plane-project-id"
              label={t(I18nKey.INTEGRATIONS$PROJECT_ID)}
              type="text"
              value={planeProjectId}
              onChange={setPlaneProjectId}
            />

            <SettingsInput
              testId="plane-module-id"
              label={t(I18nKey.INTEGRATIONS$PLANE_MODULE_ID)}
              type="text"
              value={planeModuleId}
              onChange={setPlaneModuleId}
              placeholder={t(I18nKey.INTEGRATIONS$PLANE_MODULE_ID_PLACEHOLDER)}
            />

            <div className="flex flex-col gap-1">
              <SettingsInput
                testId="plane-api-key"
                label={t(I18nKey.INTEGRATIONS$API_KEY)}
                type="password"
                value={planeApiKey}
                onChange={setPlaneApiKey}
                placeholder={t(I18nKey.INTEGRATIONS$API_KEY_PLACEHOLDER)}
              />
              {plane.apiKeyIsSet && (
                <Typography.Text
                  className="text-xs text-[var(--oh-muted)]"
                  testId="plane-api-key-set"
                >
                  {t(I18nKey.INTEGRATIONS$API_KEY_SET)}
                </Typography.Text>
              )}
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              <BrandButton
                type="button"
                variant="primary"
                testId="plane-save"
                onClick={() => void handleSavePlane()}
                isDisabled={
                  isSaving || isSavingSecret || !selectedWorkspaceId
                }
              >
                {t(I18nKey.INTEGRATIONS$SAVE)}
              </BrandButton>
              <BrandButton
                type="button"
                variant="secondary"
                testId="plane-test"
                onClick={() => void handleTestPlane()}
                isDisabled={
                  isTestingPlane || !planeEnabled || !selectedWorkspaceId
                }
              >
                {t(I18nKey.INTEGRATIONS$TEST_CONNECTION)}
              </BrandButton>
            </div>
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
              {dependencyTrack.apiKeyIsSet && (
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
                onClick={() => void handleSaveDependencyTrack()}
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
                onClick={() => void handleTestDependencyTrack()}
                isDisabled={
                  isDtTesting || !dtEnabled || !selectedWorkspaceId
                }
              >
                {t(I18nKey.INTEGRATIONS$TEST_CONNECTION)}
              </BrandButton>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default IntegrationsSettingsScreen;
