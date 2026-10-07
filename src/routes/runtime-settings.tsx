import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useConversationWorkspace } from "#/hooks/query/use-conversation-workspace";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { AppSettingsInputsSkeleton } from "#/components/features/settings/app-settings/app-settings-inputs-skeleton";
import { I18nKey } from "#/i18n/declaration";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import type { ConversationRuntimeSettings } from "#/types/settings";
import { getWorkspaceModeI18nKey } from "#/utils/workspace-mode";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

const LAST_USED_WORKSPACE_MODE_KEY = "__last_used__";
// Agent-server defaults (config.py), shown as placeholders.
const DEFAULT_MEMORY = "4g";
const DEFAULT_IMAGE = "ghcr.io/openhands/agent-server:latest-python";
const DEFAULT_WORKTREE_ROOT = "/tmp/conversation-worktrees";

interface Draft {
  mode: WorkspaceMode | null;
  runtime: ConversationRuntimeSettings;
}

const toNumber = (value: string) =>
  value.trim() === "" ? null : Number(value);
const toText = (value: string) => value.trim() || null;

function Section({
  title,
  description,
  testId,
  children,
}: {
  title: string;
  description: string;
  testId: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-t border-border pt-6 mt-2" data-testid={testId}>
      <h3 className="text-lg font-medium mb-2">{title}</h3>
      <p className="mb-4 text-sm leading-5 text-tertiary-light">
        {description}
      </p>
      <div className="flex flex-col gap-5">{children}</div>
    </div>
  );
}

function Hint({ children, testId }: { children: string; testId?: string }) {
  return (
    <p
      className="-mt-3 text-xs leading-4 text-tertiary-light"
      data-testid={testId}
    >
      {children}
    </p>
  );
}

export function RuntimeSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading } = useSettings();
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { isolated, dockerSelectable, runtimeSettingsApplied } =
    useConversationWorkspace();
  const dockerAvailable = isolated || dockerSelectable;

  const stored = React.useMemo<Draft>(
    () => ({
      mode: settings?.default_workspace_mode ?? null,
      runtime: settings?.runtime_settings ?? {},
    }),
    [settings?.default_workspace_mode, settings?.runtime_settings],
  );
  const [draft, setDraft] = React.useState<Draft | null>(null);
  const current = draft ?? stored;
  const runtime = current.runtime;
  const isDirty =
    draft !== null && JSON.stringify(draft) !== JSON.stringify(stored);
  const setRuntime = (patch: ConversationRuntimeSettings) =>
    setDraft({ ...current, runtime: { ...runtime, ...patch } });

  const modeItems = React.useMemo(() => {
    const modes: WorkspaceMode[] = ["local_repo", "new_worktree"];
    if (dockerSelectable || stored.mode === "docker_container") {
      modes.push("docker_container");
    }
    return [
      {
        key: LAST_USED_WORKSPACE_MODE_KEY,
        label: t(I18nKey.SETTINGS$WORKSPACE_MODE_LAST_USED),
      },
      ...modes.map((mode) => ({
        key: mode,
        label: t(getWorkspaceModeI18nKey(mode, "local")),
      })),
    ];
  }, [dockerSelectable, stored.mode, t]);

  const serverDefault = t(I18nKey.SETTINGS$RUNTIME_SERVER_DEFAULT);
  const previewHint = t(I18nKey.SETTINGS$RUNTIME_PREVIEW_HINT);

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    saveSettings(
      { default_workspace_mode: current.mode, runtime_settings: runtime },
      {
        onSuccess: () => displaySuccessToast(t(I18nKey.SETTINGS$SAVED)),
        onError: (error) =>
          displayErrorToast(
            retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
          ),
        onSettled: () => setDraft(null),
      },
    );
  };

  if (!settings || isLoading) return <AppSettingsInputsSkeleton />;

  return (
    <form
      data-testid="runtime-settings-screen"
      onSubmit={onSubmit}
      className="flex flex-col gap-6"
    >
      {!runtimeSettingsApplied && (
        <p
          className="rounded-lg border border-border px-4 py-3 text-sm text-tertiary-light"
          data-testid="runtime-settings-not-applied"
        >
          {t(I18nKey.SETTINGS$RUNTIME_NOT_APPLIED)}
        </p>
      )}

      <Section
        testId="conversation-workspace-settings"
        title={t(I18nKey.SETTINGS$CONVERSATION_WORKSPACE)}
        description={t(I18nKey.SETTINGS$CONVERSATION_WORKSPACE_DESCRIPTION)}
      >
        {!isolated && (
          <SettingsDropdownInput
            testId="default-workspace-mode-input"
            name="default-workspace-mode-input"
            label={t(I18nKey.SETTINGS$DEFAULT_WORKSPACE_MODE)}
            items={modeItems}
            selectedKey={current.mode ?? LAST_USED_WORKSPACE_MODE_KEY}
            onSelectionChange={(key) => {
              const value = key?.toString();
              setDraft({
                ...current,
                mode:
                  !value || value === LAST_USED_WORKSPACE_MODE_KEY
                    ? null
                    : (value as WorkspaceMode),
              });
            }}
          />
        )}
      </Section>

      <Section
        testId="docker-runtime-settings"
        title={t(I18nKey.SETTINGS$RUNTIME_DOCKER_TITLE)}
        description={t(I18nKey.SETTINGS$RUNTIME_DOCKER_DESCRIPTION)}
      >
        <p
          className="text-sm leading-5 text-tertiary-light"
          data-testid="docker-runtime-status"
        >
          {t(
            isolated
              ? I18nKey.SETTINGS$DOCKER_RUNTIME_ONLY
              : dockerSelectable
                ? I18nKey.SETTINGS$DOCKER_RUNTIME_AVAILABLE
                : I18nKey.SETTINGS$DOCKER_RUNTIME_UNAVAILABLE,
          )}
        </p>
        <SettingsInput
          testId="docker-retention-days-input"
          label={t(I18nKey.SETTINGS$RUNTIME_DOCKER_RETENTION)}
          type="number"
          min={1}
          value={runtime.docker_retention_days?.toString() ?? ""}
          placeholder={serverDefault}
          isDisabled={!dockerAvailable}
          onChange={(v) => setRuntime({ docker_retention_days: toNumber(v) })}
        />
        <SettingsInput
          testId="docker-disk-budget-input"
          label={t(I18nKey.SETTINGS$RUNTIME_DOCKER_DISK_BUDGET)}
          type="number"
          min={1}
          max={99}
          value={
            runtime.docker_disk_budget
              ? Math.round(runtime.docker_disk_budget * 100).toString()
              : ""
          }
          placeholder={serverDefault}
          isDisabled={!dockerAvailable}
          onChange={(v) => {
            const percent = toNumber(v);
            setRuntime({
              docker_disk_budget: percent === null ? null : percent / 100,
            });
          }}
        />
        <SettingsInput
          testId="docker-idle-stop-input"
          label={t(I18nKey.SETTINGS$RUNTIME_DOCKER_IDLE_STOP)}
          type="number"
          min={1}
          value={runtime.docker_idle_stop_minutes?.toString() ?? ""}
          placeholder={serverDefault}
          isDisabled={!dockerAvailable}
          onChange={(v) =>
            setRuntime({ docker_idle_stop_minutes: toNumber(v) })
          }
        />
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          <SettingsInput
            testId="docker-memory-input"
            label={t(I18nKey.SETTINGS$RUNTIME_DOCKER_MEMORY)}
            type="text"
            value={runtime.docker_memory ?? ""}
            placeholder={DEFAULT_MEMORY}
            isDisabled={!dockerAvailable}
            onChange={(v) => setRuntime({ docker_memory: toText(v) })}
          />
          <SettingsInput
            testId="docker-cpus-input"
            label={t(I18nKey.SETTINGS$RUNTIME_DOCKER_CPUS)}
            type="number"
            min={0.5}
            step={0.5}
            value={runtime.docker_cpus?.toString() ?? ""}
            placeholder="2"
            isDisabled={!dockerAvailable}
            onChange={(v) => setRuntime({ docker_cpus: toNumber(v) })}
          />
        </div>
        <SettingsSwitch
          testId="docker-browser-switch"
          isToggled={runtime.docker_browser !== false}
          isDisabled={!dockerAvailable}
          onToggle={(on) => setRuntime({ docker_browser: on ? null : false })}
        >
          {t(I18nKey.SETTINGS$RUNTIME_DOCKER_BROWSER)}
        </SettingsSwitch>
        <SettingsInput
          testId="docker-image-input"
          label={t(I18nKey.SETTINGS$RUNTIME_DOCKER_IMAGE)}
          type="text"
          value={runtime.docker_image ?? ""}
          placeholder={DEFAULT_IMAGE}
          isDisabled={!dockerAvailable}
          onChange={(v) => setRuntime({ docker_image: toText(v) })}
        />
        <Hint testId="docker-image-preview">{previewHint}</Hint>
      </Section>

      <Section
        testId="worktree-runtime-settings"
        title={t(I18nKey.SETTINGS$RUNTIME_WORKTREE_TITLE)}
        description={t(I18nKey.SETTINGS$RUNTIME_WORKTREE_DESCRIPTION)}
      >
        <SettingsDropdownInput
          testId="worktree-base-input"
          name="worktree-base-input"
          label={t(I18nKey.SETTINGS$RUNTIME_WORKTREE_BASE)}
          items={[
            {
              key: "default_branch",
              label: t(I18nKey.SETTINGS$RUNTIME_WORKTREE_BASE_DEFAULT),
            },
            {
              key: "head",
              label: t(I18nKey.SETTINGS$RUNTIME_WORKTREE_BASE_HEAD),
            },
          ]}
          selectedKey={runtime.worktree_base ?? "default_branch"}
          onSelectionChange={(key) =>
            setRuntime({
              worktree_base: key === "head" ? "head" : "default_branch",
            })
          }
        />
        <SettingsInput
          testId="worktree-retention-days-input"
          label={t(I18nKey.SETTINGS$RUNTIME_WORKTREE_RETENTION)}
          type="number"
          min={1}
          value={runtime.worktree_retention_days?.toString() ?? ""}
          placeholder={t(I18nKey.SETTINGS$RUNTIME_KEEP_FOREVER)}
          onChange={(v) => setRuntime({ worktree_retention_days: toNumber(v) })}
        />
        <SettingsSwitch
          testId="worktree-remove-on-delete-switch"
          isToggled={!!runtime.worktree_remove_on_delete}
          onToggle={(on) => setRuntime({ worktree_remove_on_delete: on })}
        >
          {t(I18nKey.SETTINGS$RUNTIME_WORKTREE_REMOVE_ON_DELETE)}
        </SettingsSwitch>
        <SettingsSwitch
          testId="worktree-delete-branch-switch"
          isToggled={!!runtime.worktree_delete_branch}
          onToggle={(on) => setRuntime({ worktree_delete_branch: on })}
        >
          {t(I18nKey.SETTINGS$RUNTIME_WORKTREE_DELETE_BRANCH)}
        </SettingsSwitch>
        <SettingsInput
          testId="worktree-location-input"
          label={t(I18nKey.SETTINGS$RUNTIME_WORKTREE_LOCATION)}
          type="text"
          value={runtime.worktree_location ?? ""}
          placeholder={DEFAULT_WORKTREE_ROOT}
          onChange={(v) => setRuntime({ worktree_location: toText(v) })}
        />
        <Hint testId="worktree-location-preview">{previewHint}</Hint>
      </Section>

      <div className="flex justify-start">
        <BrandButton
          testId="submit-button"
          variant="primary"
          type="submit"
          isDisabled={isPending || !isDirty}
        >
          {!isPending && t(I18nKey.SETTINGS$SAVE_CHANGES)}
          {isPending && t(I18nKey.SETTINGS$SAVING)}
        </BrandButton>
      </div>
    </form>
  );
}

export default RuntimeSettingsScreen;
