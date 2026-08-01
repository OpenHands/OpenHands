import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import {
  useExportSkinConfiguration,
  useInstallSkin,
  usePullSkin,
  usePushSkin,
  useSetSkinAutoPush,
  useSkinStatus,
  useUninstallSkin,
} from "#/hooks/query/use-skin";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/**
 * Skin settings: install a skin from a GitHub repo URL, pull/push the skin
 * repo, toggle auto-push, export this instance's configuration into the
 * skin, and — right after install — a single-page setup collecting every
 * secret the skin declares (names only; the user supplies the values).
 */
export function SkinSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { data: status, isLoading } = useSkinStatus();

  const installSkin = useInstallSkin();
  const uninstallSkin = useUninstallSkin();
  const pullSkin = usePullSkin();
  const pushSkin = usePushSkin();
  const setAutoPush = useSetSkinAutoPush();
  const exportConfiguration = useExportSkinConfiguration();
  const createSecret = useCreateSecret();

  const [repoUrl, setRepoUrl] = React.useState("");
  const [ref, setRef] = React.useState("");
  const [autoPush, setAutoPushChecked] = React.useState(true);
  const [showSecretSetup, setShowSecretSetup] = React.useState(false);
  const [secretValues, setSecretValues] = React.useState<
    Record<string, string>
  >({});

  if (isLoading) return null;

  const onInstall = () => {
    installSkin.mutate(
      { repoUrl, ref: ref || undefined, autoPush },
      {
        onSuccess: (result) => {
          displaySuccessToast(t(I18nKey.SKIN$INSTALL_SUCCESS));
          if ((result.secrets?.length ?? 0) > 0) {
            setShowSecretSetup(true);
          }
        },
        onError: (error) => displayErrorToast(error.message),
      },
    );
  };

  const onSaveSecrets = async () => {
    const entries = Object.entries(secretValues).filter(([, v]) => v);
    try {
      await Promise.all(
        entries.map(([name, value]) =>
          createSecret.mutateAsync({ name, value }),
        ),
      );
      displaySuccessToast(t(I18nKey.SKIN$SECRETS_SAVED));
      setShowSecretSetup(false);
      setSecretValues({});
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
    }
  };

  // ── one-page guided secret setup ─────────────────────────────────────────
  if (status?.installed && showSecretSetup && (status.secrets?.length ?? 0)) {
    return (
      <div
        data-testid="skin-secret-setup"
        className="flex flex-col gap-6 max-w-xl"
      >
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.SKIN$SECRET_SETUP_DESCRIPTION)}
        </p>
        {(status.secrets ?? []).map((secret) => (
          <SettingsInput
            key={secret.name}
            testId={`skin-secret-${secret.name}`}
            label={secret.name}
            hint={secret.description}
            type="password"
            value={secretValues[secret.name] ?? ""}
            onChange={(value) =>
              setSecretValues((prev) => ({ ...prev, [secret.name]: value }))
            }
          />
        ))}
        <div className="flex gap-2">
          <BrandButton
            testId="skin-secrets-save"
            variant="primary"
            type="button"
            isDisabled={createSecret.isPending}
            onClick={onSaveSecrets}
          >
            {t(I18nKey.SKIN$SAVE_SECRETS)}
          </BrandButton>
          <BrandButton
            variant="secondary"
            type="button"
            onClick={() => setShowSecretSetup(false)}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
        </div>
      </div>
    );
  }

  // ── installed skin management ────────────────────────────────────────────
  if (status?.installed) {
    return (
      <div data-testid="skin-settings" className="flex flex-col gap-6 max-w-xl">
        <div className="flex flex-col gap-1 text-sm">
          <span className="font-semibold">{status.name}</span>
          <span className="text-[var(--oh-muted)]">{status.repoUrl}</span>
          {status.branch ? (
            <span className="text-[var(--oh-muted)]">
              {t(I18nKey.SKIN$BRANCH_LABEL)}: {status.branch}
            </span>
          ) : null}
          {status.error ? (
            <span className="text-red-400">{status.error}</span>
          ) : null}
        </div>

        <SettingsSwitch
          testId="skin-auto-push-switch"
          isToggled={!!status.autoPush}
          onToggle={(value) => setAutoPush.mutate(value)}
        >
          {t(I18nKey.SKIN$AUTO_PUSH_LABEL)}
        </SettingsSwitch>

        <div className="flex flex-wrap gap-2">
          <BrandButton
            testId="skin-pull-button"
            variant="secondary"
            type="button"
            isDisabled={pullSkin.isPending}
            onClick={() =>
              pullSkin.mutate(undefined, {
                onSuccess: () =>
                  displaySuccessToast(t(I18nKey.SKIN$PULL_SUCCESS)),
                onError: (error) => displayErrorToast(error.message),
              })
            }
          >
            {t(I18nKey.SKIN$PULL)}
          </BrandButton>
          <BrandButton
            testId="skin-push-button"
            variant="secondary"
            type="button"
            isDisabled={pushSkin.isPending}
            onClick={() =>
              pushSkin.mutate(undefined, {
                onSuccess: (result) => {
                  if (result.pullRequest) {
                    displaySuccessToast(
                      `${t(I18nKey.SKIN$PUSH_OPENED_PR)}: ${result.pullRequest.url}`,
                    );
                  } else {
                    displaySuccessToast(t(I18nKey.SKIN$PUSH_SUCCESS));
                  }
                },
                onError: (error) => displayErrorToast(error.message),
              })
            }
          >
            {t(I18nKey.SKIN$PUSH)}
          </BrandButton>
          <BrandButton
            testId="skin-export-button"
            variant="secondary"
            type="button"
            isDisabled={exportConfiguration.isPending}
            onClick={() =>
              exportConfiguration.mutate(undefined, {
                onSuccess: () =>
                  displaySuccessToast(t(I18nKey.SKIN$EXPORT_SUCCESS)),
                onError: (error) => displayErrorToast(error.message),
              })
            }
          >
            {t(I18nKey.SKIN$EXPORT)}
          </BrandButton>
          <BrandButton
            testId="skin-uninstall-button"
            variant="danger"
            type="button"
            isDisabled={uninstallSkin.isPending}
            onClick={() =>
              uninstallSkin.mutate(undefined, {
                onSuccess: () =>
                  displaySuccessToast(t(I18nKey.SKIN$UNINSTALL_SUCCESS)),
                onError: (error) => displayErrorToast(error.message),
              })
            }
          >
            {t(I18nKey.SKIN$UNINSTALL)}
          </BrandButton>
        </div>

        {(status.secrets?.length ?? 0) > 0 ? (
          <BrandButton
            testId="skin-setup-secrets-button"
            variant="secondary"
            type="button"
            className="self-start"
            onClick={() => setShowSecretSetup(true)}
          >
            {t(I18nKey.SKIN$SETUP_SECRETS)}
          </BrandButton>
        ) : null}
      </div>
    );
  }

  // ── install form ─────────────────────────────────────────────────────────
  return (
    <div
      data-testid="skin-install-form"
      className="flex flex-col gap-6 max-w-xl"
    >
      <p className="text-sm text-[var(--oh-muted)]">
        {t(I18nKey.SKIN$INSTALL_DESCRIPTION)}
      </p>
      <SettingsInput
        testId="skin-repo-url-input"
        label={t(I18nKey.SKIN$REPO_URL_LABEL)}
        placeholder={t(I18nKey.SKIN$REPO_URL_PLACEHOLDER)}
        type="url"
        value={repoUrl}
        onChange={setRepoUrl}
      />
      <SettingsInput
        testId="skin-ref-input"
        label={t(I18nKey.SKIN$REF_LABEL)}
        showOptionalTag
        type="text"
        value={ref}
        onChange={setRef}
      />
      <SettingsSwitch
        testId="skin-install-auto-push-switch"
        isToggled={autoPush}
        onToggle={setAutoPushChecked}
      >
        {t(I18nKey.SKIN$AUTO_PUSH_LABEL)}
      </SettingsSwitch>
      <BrandButton
        testId="skin-install-button"
        variant="primary"
        type="button"
        className="self-start"
        isDisabled={!repoUrl || installSkin.isPending}
        onClick={onInstall}
      >
        {t(I18nKey.SKIN$INSTALL)}
      </BrandButton>
    </div>
  );
}

export default SkinSettingsScreen;
