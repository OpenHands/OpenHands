import { useState, type FormEvent } from "react";
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
import { getErrorStatus } from "#/hooks/query/use-settings";
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

  // `null` means untouched, so the switch keeps following the server state
  // (including a status refetch) until the operator actually flips it.
  const [enabledOverride, setEnabledOverride] = useState<boolean | null>(null);
  const enabled = enabledOverride ?? status.enabled;
  const enabledHasChanged =
    enabledOverride !== null && enabled !== status.enabled;

  const [tokenText, setTokenText] = useState("");
  const [clearToken, setClearToken] = useState(false);
  const [encryptionKeyText, setEncryptionKeyText] = useState("");
  const [clearEncryptionKey, setClearEncryptionKey] = useState(false);

  const tokenHasChanged = tokenText.trim().length > 0 || clearToken;
  const encryptionKeyHasChanged =
    encryptionKeyText.trim().length > 0 || clearEncryptionKey;

  const formIsClean =
    !enabledHasChanged &&
    !intervalHasChanged &&
    !repoUrlHasChanged &&
    !branchHasChanged &&
    !pathHasChanged &&
    !authorNameHasChanged &&
    !authorEmailHasChanged &&
    !tokenHasChanged &&
    !encryptionKeyHasChanged;

  const resetChangeFlags = () => {
    setEnabledOverride(null);
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

  // The secret inputs are remounted -- and therefore emptied -- whenever the
  // matching clear switch flips, so drop the typed value with them. Keeping it
  // in state would leave the field marked dirty against an empty input and
  // submit `token: ""`, an override that fails every later push with
  // `fatal: Authentication failed`.
  const toggleClearToken = (isToggled: boolean) => {
    setClearToken(isToggled);
    setTokenText("");
  };

  const toggleClearEncryptionKey = (isToggled: boolean) => {
    setClearEncryptionKey(isToggled);
    setEncryptionKeyText("");
  };

  // A cleared field posts `null` -- clear the override and fall back to the
  // environment default -- rather than `""`, which the backend stored as a
  // literal empty override. An empty branch or path then made the next git
  // command fatal (`git checkout -B ""`, `git add -A -- ""`), wedging every
  // subsequent sync cycle with that error in the status banner.
  const clearedFieldAsNull = (formData: FormData, name: string) =>
    formData.get(name)?.toString().trim() || null;

  // A plain submit handler rather than `<form action={...}>`: React resets an
  // uncontrolled form as soon as the action returns, which wiped every edit
  // while the save was still in flight -- so a rejected save left the operator
  // with the old values and nothing to retry.
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const body: GitSyncConfigUpdateRequest = {};

    if (enabledHasChanged) {
      body.enabled = enabled;
    }
    if (intervalHasChanged) {
      // Blank reads as manual-only rather than clearing the override, so an
      // emptied field can never be mistaken for "keep syncing on a timer".
      const raw = formData.get("git-sync-interval-input")?.toString().trim();
      const parsed = Number(raw);
      body.interval_seconds =
        raw && Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : 0;
    }
    if (repoUrlHasChanged) {
      body.repo_url = clearedFieldAsNull(formData, "git-sync-repo-url-input");
    }
    if (branchHasChanged) {
      body.branch = clearedFieldAsNull(formData, "git-sync-branch-input");
    }
    if (pathHasChanged) {
      body.path = clearedFieldAsNull(formData, "git-sync-path-input");
    }
    if (authorNameHasChanged) {
      body.author_name = clearedFieldAsNull(
        formData,
        "git-sync-author-name-input",
      );
    }
    if (authorEmailHasChanged) {
      body.author_email = clearedFieldAsNull(
        formData,
        "git-sync-author-email-input",
      );
    }
    // The secrets come from state rather than FormData: the clear switch
    // remounts (and empties) their inputs, so the two disagree for a render
    // and only state knows what the operator actually typed.
    const token = tokenText.trim();
    const encryptionKey = encryptionKeyText.trim();
    if (clearToken) {
      body.token = null;
    } else if (token) {
      body.token = token;
    }
    if (clearEncryptionKey) {
      body.encryption_key = null;
    } else if (encryptionKey) {
      body.encryption_key = encryptionKey;
    }

    updateConfig(body, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATIONS$GIT_SYNC$CONFIG_SAVED));
        // Only on success: clearing the flags after a failure would disable
        // Save and silently un-toggle the clear switches, leaving no way to
        // retry the change that was just rejected.
        resetChangeFlags();
      },
      onError: (error) => {
        displayErrorToast(
          // 409 is the backend refusing to enable sync in a deployment that
          // booted with it off -- a restart with the env var set, not a
          // transient failure the operator should retry.
          getErrorStatus(error) === 409
            ? t(I18nKey.AUTOMATIONS$GIT_SYNC$ENABLE_BLOCKED_ERROR)
            : getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
        );
      },
    });
  };

  return (
    <SectionCard
      icon={<CogIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$GIT_SYNC$CONFIG_TITLE)}
    >
      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <SettingsSwitch
            testId="git-sync-enabled-switch"
            isToggled={enabled}
            isDisabled={!canManage}
            onToggle={setEnabledOverride}
          >
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_ENABLED)}
          </SettingsSwitch>
          <p className="text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$ENABLED_HELP)}
          </p>
        </div>

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

        <div className="flex flex-col gap-2">
          <SettingsInput
            testId="git-sync-author-name-input"
            name="git-sync-author-name-input"
            type="text"
            label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_AUTHOR_NAME)}
            showOptionalTag
            defaultValue=""
            isDisabled={!canManage}
            // Dirty on any edit, not just a non-empty one: gating on the value
            // meant emptying the field never posted `author_name: null`, so a
            // wrong override could never be cleared back to the default.
            onChange={() => setAuthorNameHasChanged(true)}
          />

          <SettingsInput
            testId="git-sync-author-email-input"
            name="git-sync-author-email-input"
            type="email"
            label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_AUTHOR_EMAIL)}
            showOptionalTag
            defaultValue=""
            isDisabled={!canManage}
            onChange={() => setAuthorEmailHasChanged(true)}
          />
          <p className="text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$GIT_SYNC$AUTHOR_HELP)}
          </p>
        </div>

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
            onToggle={toggleClearToken}
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
            onToggle={toggleClearEncryptionKey}
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
