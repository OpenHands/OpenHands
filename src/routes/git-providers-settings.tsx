import { useQueryClient } from "@tanstack/react-query";
import React from "react";
import { useTranslation } from "react-i18next";
import { Navigate } from "react-router";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import { useDeleteSecret } from "#/hooks/mutation/use-delete-secret";
import { useSettings } from "#/hooks/query/use-settings";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  GIT_PROVIDER_AUTH_METHODS,
  GIT_PROVIDER_ID_PATTERN,
  type GitProviderAuthMethod,
  type GitProviderPreference,
} from "#/types/git-provider";
import {
  allGitProviderSecretNames,
  gitProviderPasswordSecretName,
  gitProviderSshPrivateKeySecretName,
  gitProviderTokenSecretName,
  gitProviderUsernameSecretName,
} from "#/utils/git-provider-secrets";
import { cn } from "#/utils/utils";
import {
  settingsListScrollContainerClassName,
  settingsListTableHeadClassName,
  settingsListTableHeaderCellClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const handle = { hideTitle: true };

type EditorMode = "list" | "add" | "edit";

function authMethodLabelKey(method: GitProviderAuthMethod): I18nKey {
  switch (method) {
    case "pat":
      return I18nKey.GIT_PROVIDERS$AUTH_PAT;
    case "password":
      return I18nKey.GIT_PROVIDERS$AUTH_PASSWORD;
    case "ssh":
      return I18nKey.GIT_PROVIDERS$AUTH_SSH;
    default:
      return I18nKey.GIT_PROVIDERS$AUTH_PAT;
  }
}

function secretConfigured(secretNames: Set<string>, name: string): boolean {
  return secretNames.has(name);
}

export function GitProvidersSettingsScreen() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: settings, isLoading } = useSettings();
  const { data: secrets } = useSearchSecrets();
  const { mutateAsync: saveSettings, isPending: isSaving } = useSaveSettings();
  const { mutateAsync: createSecret } = useCreateSecret();
  const { mutateAsync: deleteSecret } = useDeleteSecret();

  const invalidateSecrets = () => {
    queryClient.invalidateQueries({ queryKey: ["secrets-search"] });
    queryClient.invalidateQueries({ queryKey: ["secrets"] });
  };

  const [view, setView] = React.useState<EditorMode>("list");
  const [editingId, setEditingId] = React.useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = React.useState<string | null>(null);

  const [formId, setFormId] = React.useState("");
  const [formLabel, setFormLabel] = React.useState("");
  const [formHost, setFormHost] = React.useState("");
  const [formAuthMethod, setFormAuthMethod] =
    React.useState<GitProviderAuthMethod>("pat");
  const [tokenValue, setTokenValue] = React.useState("");
  const [usernameValue, setUsernameValue] = React.useState("");
  const [passwordValue, setPasswordValue] = React.useState("");
  const [sshKeyValue, setSshKeyValue] = React.useState("");

  const providers = settings?.git_providers ?? [];
  const secretNameSet = React.useMemo(
    () => new Set((secrets ?? []).map((s) => s.name)),
    [secrets],
  );

  if (backend.kind !== "local") {
    return <Navigate to="/settings/app" replace />;
  }

  const resetForm = () => {
    setFormId("");
    setFormLabel("");
    setFormHost("");
    setFormAuthMethod("pat");
    setTokenValue("");
    setUsernameValue("");
    setPasswordValue("");
    setSshKeyValue("");
    setEditingId(null);
  };

  const openAdd = () => {
    resetForm();
    setView("add");
  };

  const openEdit = (provider: GitProviderPreference) => {
    setEditingId(provider.id);
    setFormId(provider.id);
    setFormLabel(provider.label);
    setFormHost(provider.host);
    setFormAuthMethod(provider.auth_method);
    setTokenValue("");
    setUsernameValue("");
    setPasswordValue("");
    setSshKeyValue("");
    setView("edit");
  };

  const persistProviders = async (
    next: GitProviderPreference[],
  ): Promise<void> => {
    await saveSettings({ git_providers: next });
  };

  const writeSecretsForProvider = async (providerId: string) => {
    const tasks: Promise<void>[] = [];
    if (tokenValue.trim()) {
      tasks.push(
        createSecret({
          name: gitProviderTokenSecretName(providerId),
          value: tokenValue.trim(),
          description: t(I18nKey.GIT_PROVIDERS$SECRET_TOKEN_DESC, {
            id: providerId,
          }),
        }),
      );
    }
    if (usernameValue.trim()) {
      tasks.push(
        createSecret({
          name: gitProviderUsernameSecretName(providerId),
          value: usernameValue.trim(),
          description: t(I18nKey.GIT_PROVIDERS$SECRET_USERNAME_DESC, {
            id: providerId,
          }),
        }),
      );
    }
    if (passwordValue.trim()) {
      tasks.push(
        createSecret({
          name: gitProviderPasswordSecretName(providerId),
          value: passwordValue.trim(),
          description: t(I18nKey.GIT_PROVIDERS$SECRET_PASSWORD_DESC, {
            id: providerId,
          }),
        }),
      );
    }
    if (sshKeyValue.trim()) {
      tasks.push(
        createSecret({
          name: gitProviderSshPrivateKeySecretName(providerId),
          value: sshKeyValue.trim(),
          description: t(I18nKey.GIT_PROVIDERS$SECRET_SSH_DESC, {
            id: providerId,
          }),
        }),
      );
    }
    await Promise.all(tasks);
  };

  const handleSaveProvider = async () => {
    const id = formId.trim();
    const label = formLabel.trim();
    const host = formHost.trim();

    if (!GIT_PROVIDER_ID_PATTERN.test(id)) {
      displayErrorToast(t(I18nKey.GIT_PROVIDERS$INVALID_ID));
      return;
    }
    if (!label || !host) {
      displayErrorToast(t(I18nKey.GIT_PROVIDERS$REQUIRED_FIELDS));
      return;
    }
    if (view === "add" && providers.some((p) => p.id === id)) {
      displayErrorToast(t(I18nKey.GIT_PROVIDERS$DUPLICATE_ID));
      return;
    }

    const entry: GitProviderPreference = {
      id,
      label,
      host,
      auth_method: formAuthMethod,
    };
    const next =
      view === "add"
        ? [...providers, entry]
        : providers.map((p) => (p.id === editingId ? entry : p));

    try {
      await persistProviders(next);
      await writeSecretsForProvider(id);
      invalidateSecrets();
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      resetForm();
      setView("list");
    } catch (error) {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    const next = providers.filter((p) => p.id !== deleteTarget);
    try {
      await persistProviders(next);
      await Promise.all(
        allGitProviderSecretNames(deleteTarget).map((name) =>
          deleteSecret(name).catch(() => undefined),
        ),
      );
      invalidateSecrets();
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
    } catch (error) {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
      );
    } finally {
      setDeleteTarget(null);
    }
  };

  const authMethodItems = GIT_PROVIDER_AUTH_METHODS.map((method) => ({
    key: method,
    label: t(authMethodLabelKey(method)),
  }));

  const isFormView = view === "add" || view === "edit";

  return (
    <div
      data-testid="git-providers-settings-screen"
      className="flex flex-col gap-6"
    >
      {view === "list" ? (
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 space-y-1">
            <Typography.H2>
              {t(I18nKey.SETTINGS$NAV_GIT_PROVIDERS)}
            </Typography.H2>
            <p
              data-testid="settings-page-subtitle"
              className="text-sm leading-5 text-tertiary-light"
            >
              {t(I18nKey.SETTINGS$PAGE_GIT_PROVIDERS_SUBLINE)}
            </p>
          </div>
          <BrandButton
            testId="add-git-provider-button"
            type="button"
            variant="primary"
            className="shrink-0 whitespace-nowrap"
            onClick={openAdd}
            isDisabled={isLoading}
          >
            {t(I18nKey.GIT_PROVIDERS$ADD)}
          </BrandButton>
        </div>
      ) : null}

      {isFormView ? (
        <div className="flex flex-col gap-2">
          <BackNavButton
            testId="back-to-git-providers"
            onClick={() => {
              resetForm();
              setView("list");
            }}
          >
            {t(I18nKey.BUTTON$BACK)}
          </BackNavButton>
          <Typography.H2 testId="git-provider-editor-title">
            {view === "add"
              ? t(I18nKey.GIT_PROVIDERS$ADD)
              : t(I18nKey.GIT_PROVIDERS$EDIT)}
          </Typography.H2>
        </div>
      ) : null}

      {view === "list" && !isLoading && providers.length === 0 && (
        <div
          data-testid="git-providers-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.GIT_PROVIDERS$EMPTY)}
          </p>
        </div>
      )}

      {view === "list" && providers.length > 0 && (
        <div className={settingsListScrollContainerClassName}>
          <table className="w-full min-w-full table-fixed">
            <thead className={settingsListTableHeadClassName}>
              <tr>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/4")}
                >
                  {t(I18nKey.SETTINGS$NAME)}
                </th>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/4")}
                >
                  {t(I18nKey.GIT_PROVIDERS$HOST)}
                </th>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/4")}
                >
                  {t(I18nKey.GIT_PROVIDERS$AUTH_METHOD)}
                </th>
                <th
                  className={cn(
                    settingsListTableHeaderCellClassName,
                    "w-1/4 text-right",
                  )}
                >
                  {t(I18nKey.SETTINGS$ACTIONS)}
                </th>
              </tr>
            </thead>
            <tbody>
              {providers.map((provider) => {
                const hasCreds =
                  secretConfigured(
                    secretNameSet,
                    gitProviderTokenSecretName(provider.id),
                  ) ||
                  (secretConfigured(
                    secretNameSet,
                    gitProviderUsernameSecretName(provider.id),
                  ) &&
                    secretConfigured(
                      secretNameSet,
                      gitProviderPasswordSecretName(provider.id),
                    )) ||
                  secretConfigured(
                    secretNameSet,
                    gitProviderSshPrivateKeySecretName(provider.id),
                  );
                return (
                  <tr
                    key={provider.id}
                    data-testid={`git-provider-row-${provider.id}`}
                    className="border-b border-[var(--oh-border)]"
                  >
                    <td className="px-3 py-3 text-sm">
                      <div className="font-medium">{provider.label}</div>
                      <div className="text-xs text-[var(--oh-muted)]">
                        {provider.id}
                      </div>
                      <div className="text-xs text-[var(--oh-muted)] mt-1">
                        {hasCreds
                          ? t(I18nKey.GIT_PROVIDERS$CREDENTIALS_SET)
                          : t(I18nKey.GIT_PROVIDERS$CREDENTIALS_MISSING)}
                      </div>
                    </td>
                    <td className="px-3 py-3 text-sm truncate">
                      {provider.host}
                    </td>
                    <td className="px-3 py-3 text-sm">
                      {t(authMethodLabelKey(provider.auth_method))}
                    </td>
                    <td className="px-3 py-3 text-sm text-right space-x-2">
                      <BrandButton
                        type="button"
                        variant="secondary"
                        testId={`edit-git-provider-${provider.id}`}
                        onClick={() => openEdit(provider)}
                      >
                        {t(I18nKey.BUTTON$EDIT)}
                      </BrandButton>
                      <BrandButton
                        type="button"
                        variant="danger"
                        testId={`delete-git-provider-${provider.id}`}
                        onClick={() => setDeleteTarget(provider.id)}
                      >
                        {t(I18nKey.BUTTON$DELETE)}
                      </BrandButton>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {isFormView && (
        <div className="flex flex-col gap-6 max-w-xl">
          <SettingsInput
            testId="git-provider-id-input"
            label={t(I18nKey.GIT_PROVIDERS$ID)}
            type="text"
            value={formId}
            onChange={setFormId}
            isDisabled={view === "edit"}
            hint={t(I18nKey.GIT_PROVIDERS$ID_HINT)}
            pattern="[A-Za-z][A-Za-z0-9_]{0,31}"
            required
          />
          <SettingsInput
            testId="git-provider-label-input"
            label={t(I18nKey.GIT_PROVIDERS$LABEL)}
            type="text"
            value={formLabel}
            onChange={setFormLabel}
            required
          />
          <SettingsInput
            testId="git-provider-host-input"
            label={t(I18nKey.GIT_PROVIDERS$HOST)}
            type="text"
            value={formHost}
            onChange={setFormHost}
            placeholder={t(I18nKey.GIT_PROVIDERS$HOST_PLACEHOLDER)}
            required
          />
          <SettingsDropdownInput
            testId="git-provider-auth-method-input"
            name="git-provider-auth-method"
            label={t(I18nKey.GIT_PROVIDERS$AUTH_METHOD)}
            items={authMethodItems}
            selectedKey={formAuthMethod}
            onSelectionChange={(key) => {
              const value = key?.toString() as
                | GitProviderAuthMethod
                | undefined;
              if (value && GIT_PROVIDER_AUTH_METHODS.includes(value)) {
                setFormAuthMethod(value);
              }
            }}
          />

          {(formAuthMethod === "pat" || formAuthMethod === "password") && (
            <>
              {formAuthMethod === "password" && (
                <SettingsInput
                  testId="git-provider-username-input"
                  label={t(I18nKey.GIT_PROVIDERS$USERNAME)}
                  type="text"
                  value={usernameValue}
                  onChange={setUsernameValue}
                  hint={
                    view === "edit"
                      ? t(I18nKey.GIT_PROVIDERS$SECRET_WRITE_ONLY_HINT)
                      : undefined
                  }
                />
              )}
              {formAuthMethod === "pat" ? (
                <SettingsInput
                  testId="git-provider-token-input"
                  label={t(I18nKey.GIT_PROVIDERS$TOKEN)}
                  type="password"
                  value={tokenValue}
                  onChange={setTokenValue}
                  hint={
                    view === "edit"
                      ? t(I18nKey.GIT_PROVIDERS$SECRET_WRITE_ONLY_HINT)
                      : undefined
                  }
                />
              ) : (
                <SettingsInput
                  testId="git-provider-password-input"
                  label={t(I18nKey.GIT_PROVIDERS$PASSWORD)}
                  type="password"
                  value={passwordValue}
                  onChange={setPasswordValue}
                  hint={
                    view === "edit"
                      ? t(I18nKey.GIT_PROVIDERS$SECRET_WRITE_ONLY_HINT)
                      : undefined
                  }
                />
              )}
            </>
          )}

          {formAuthMethod === "ssh" && (
            <div className="flex flex-col gap-2">
              <label
                htmlFor="git-provider-ssh-key-input"
                className="text-sm font-medium"
              >
                {t(I18nKey.GIT_PROVIDERS$SSH_PRIVATE_KEY)}
              </label>
              {view === "edit" && (
                <p className="text-xs text-[var(--oh-muted)]">
                  {t(I18nKey.GIT_PROVIDERS$SECRET_WRITE_ONLY_HINT)}
                </p>
              )}
              <textarea
                id="git-provider-ssh-key-input"
                data-testid="git-provider-ssh-key-input"
                className="min-h-32 w-full rounded-md border border-[var(--oh-border-input)] bg-[var(--oh-surface)] px-3 py-2 text-sm font-mono"
                value={sshKeyValue}
                onChange={(e) => setSshKeyValue(e.target.value)}
                spellCheck={false}
              />
            </div>
          )}

          <div className="flex justify-start">
            <BrandButton
              testId="save-git-provider-button"
              type="button"
              variant="primary"
              onClick={() => void handleSaveProvider()}
              isDisabled={isSaving}
            >
              {isSaving
                ? t(I18nKey.SETTINGS$SAVING)
                : t(I18nKey.SETTINGS$SAVE_CHANGES)}
            </BrandButton>
          </div>
        </div>
      )}

      {deleteTarget && (
        <ConfirmationModal
          text={t(I18nKey.GIT_PROVIDERS$DELETE_CONFIRM)}
          onConfirm={() => void handleConfirmDelete()}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}

export default GitProvidersSettingsScreen;
