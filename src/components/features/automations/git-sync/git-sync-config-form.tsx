import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SectionCard } from "#/components/features/automations/detail/section-card";
import CogIcon from "#/icons/cog.svg?react";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { useUpdateGitSyncConfig } from "#/hooks/query/use-git-sync";
import type {
  GitSyncConfigUpdateRequest,
  GitSyncStatus,
} from "#/types/git-sync";

interface GitSyncConfigFormProps {
  status: GitSyncStatus;
  canManage: boolean;
}

export function GitSyncConfigForm({
  status,
  canManage,
}: GitSyncConfigFormProps) {
  const { t } = useTranslation("openhands");
  const { mutate: updateConfig, isPending } = useUpdateGitSyncConfig();

  const [intervalHasChanged, setIntervalHasChanged] = useState(false);
  const [repoUrlHasChanged, setRepoUrlHasChanged] = useState(false);
  const [branchHasChanged, setBranchHasChanged] = useState(false);
  const [pathHasChanged, setPathHasChanged] = useState(false);
  const [authorNameHasChanged, setAuthorNameHasChanged] = useState(false);
  const [authorEmailHasChanged, setAuthorEmailHasChanged] = useState(false);

  const [tokenText, setTokenText] = useState("");
  const [clearToken, setClearToken] = useState(false);
  const [encryptionKeyText, setEncryptionKeyText] = useState("");
  const [clearEncryptionKey, setClearEncryptionKey] = useState(false);

  const tokenHasChanged = tokenText.trim().length > 0 || clearToken;
  const encryptionKeyHasChanged =
    encryptionKeyText.trim().length > 0 || clearEncryptionKey;

  const formIsClean =
    !intervalHasChanged &&
    !repoUrlHasChanged &&
    !branchHasChanged &&
    !pathHasChanged &&
    !authorNameHasChanged &&
    !authorEmailHasChanged &&
    !tokenHasChanged &&
    !encryptionKeyHasChanged;

  const resetChangeFlags = () => {
    setIntervalHasChanged(false);
    setRepoUrlHasChanged(false);
    setBranchHasChanged(false);
    setPathHasChanged(false);
    setAuthorNameHasChanged(false);
    setAuthorEmailHasChanged(false);
    setTokenText("");
    setClearToken(false);
    setEncryptionKeyText("");
    setClearEncryptionKey(false);
  };

  const formAction = (formData: FormData) => {
    const body: GitSyncConfigUpdateRequest = {};

    if (intervalHasChanged) {
      // Blank reads as manual-only rather than clearing the override, so an
      // emptied field can never be mistaken for "keep syncing on a timer".
      const raw = formData.get("git-sync-interval-input")?.toString().trim();
      const parsed = Number(raw);
      body.interval_seconds =
        raw && Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : 0;
    }
    if (repoUrlHasChanged) {
      body.repo_url = formData
        .get("git-sync-repo-url-input")
        ?.toString()
        .trim();
    }
    if (branchHasChanged) {
      body.branch = formData.get("git-sync-branch-input")?.toString().trim();
    }
    if (pathHasChanged) {
      body.path = formData.get("git-sync-path-input")?.toString().trim();
    }
    if (authorNameHasChanged) {
      body.author_name = formData
        .get("git-sync-author-name-input")
        ?.toString()
        .trim();
    }
    if (authorEmailHasChanged) {
      body.author_email = formData
        .get("git-sync-author-email-input")
        ?.toString()
        .trim();
    }
    if (clearToken) {
      body.token = null;
    } else if (tokenText.trim()) {
      body.token = formData.get("git-sync-token-input")?.toString().trim();
    }
    if (clearEncryptionKey) {
      body.encryption_key = null;
    } else if (encryptionKeyText.trim()) {
      body.encryption_key = formData
        .get("git-sync-encryption-key-input")
        ?.toString()
        .trim();
    }

    updateConfig(body, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATIONS$GIT_SYNC$CONFIG_SAVED));
      },
      onError: (error) => {
        displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
      },
      onSettled: resetChangeFlags,
    });
  };

  return (
    <SectionCard
      icon={<CogIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$GIT_SYNC$CONFIG_TITLE)}
    >
      <form action={formAction} className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <SettingsInput
            testId="git-sync-interval-input"
            name="git-sync-interval-input"
            type="number"
            min={0}
            label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_INTERVAL)}
            defaultValue={String(status.interval_seconds)}
            isDisabled={!canManage}
            onChange={(value) =>
              setIntervalHasChanged(
                value.trim() !== String(status.interval_seconds),
              )
            }
          />
          <p className="text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$INTERVAL_HELP)}
          </p>
        </div>

        <SettingsInput
          testId="git-sync-repo-url-input"
          name="git-sync-repo-url-input"
          type="text"
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_REPO_URL)}
          defaultValue={status.repo_url}
          placeholder={t(I18nKey.AUTOMATIONS$GIT_SYNC$REPO_URL_PLACEHOLDER)}
          isDisabled={!canManage}
          onChange={(value) =>
            setRepoUrlHasChanged(value.trim() !== status.repo_url)
          }
        />

        <SettingsInput
          testId="git-sync-branch-input"
          name="git-sync-branch-input"
          type="text"
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_BRANCH)}
          defaultValue={status.branch}
          placeholder={t(I18nKey.AUTOMATIONS$GIT_SYNC$BRANCH_PLACEHOLDER)}
          isDisabled={!canManage}
          onChange={(value) =>
            setBranchHasChanged(value.trim() !== status.branch)
          }
        />

        <SettingsInput
          testId="git-sync-path-input"
          name="git-sync-path-input"
          type="text"
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_PATH)}
          defaultValue={status.path}
          placeholder={t(I18nKey.AUTOMATIONS$GIT_SYNC$PATH_PLACEHOLDER)}
          isDisabled={!canManage}
          onChange={(value) => setPathHasChanged(value.trim() !== status.path)}
        />

        <SettingsInput
          testId="git-sync-author-name-input"
          name="git-sync-author-name-input"
          type="text"
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_AUTHOR_NAME)}
          showOptionalTag
          defaultValue=""
          isDisabled={!canManage}
          onChange={(value) => setAuthorNameHasChanged(value.trim().length > 0)}
        />

        <SettingsInput
          testId="git-sync-author-email-input"
          name="git-sync-author-email-input"
          type="email"
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_AUTHOR_EMAIL)}
          showOptionalTag
          defaultValue=""
          isDisabled={!canManage}
          onChange={(value) =>
            setAuthorEmailHasChanged(value.trim().length > 0)
          }
        />

        <div className="flex flex-col gap-2">
          <SettingsInput
            key={clearToken ? "token-cleared" : "token-editable"}
            testId="git-sync-token-input"
            name="git-sync-token-input"
            type="password"
            label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_TOKEN)}
            placeholder={t(I18nKey.AUTOMATIONS$GIT_SYNC$TOKEN_PLACEHOLDER)}
            isDisabled={!canManage || clearToken}
            onChange={setTokenText}
          />
          <p className="text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$TOKEN_HELP)}
          </p>
          <SettingsSwitch
            testId="git-sync-clear-token-switch"
            isToggled={clearToken}
            isDisabled={!canManage}
            onToggle={setClearToken}
          >
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$CLEAR_TOKEN)}
          </SettingsSwitch>
        </div>

        <div className="flex flex-col gap-2">
          <SettingsInput
            key={clearEncryptionKey ? "key-cleared" : "key-editable"}
            testId="git-sync-encryption-key-input"
            name="git-sync-encryption-key-input"
            type="password"
            label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_ENCRYPTION_KEY)}
            placeholder={t(
              status.encryption_enabled
                ? I18nKey.AUTOMATIONS$GIT_SYNC$KEY_SET_PLACEHOLDER
                : I18nKey.AUTOMATIONS$GIT_SYNC$KEY_UNSET_PLACEHOLDER,
            )}
            isDisabled={!canManage || clearEncryptionKey}
            onChange={setEncryptionKeyText}
          />
          <SettingsSwitch
            testId="git-sync-clear-encryption-key-switch"
            isToggled={clearEncryptionKey}
            isDisabled={!canManage}
            onToggle={setClearEncryptionKey}
          >
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$CLEAR_ENCRYPTION_KEY)}
          </SettingsSwitch>
        </div>

        <div className="flex justify-start">
          <BrandButton
            testId="git-sync-save-button"
            variant="primary"
            type="submit"
            isDisabled={!canManage || isPending || formIsClean}
          >
            {!isPending && t(I18nKey.SETTINGS$SAVE_CHANGES)}
            {isPending && t(I18nKey.SETTINGS$SAVING)}
          </BrandButton>
        </div>
      </form>
    </SectionCard>
  );
}
