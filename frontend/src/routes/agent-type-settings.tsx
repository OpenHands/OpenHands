import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/** Map from the user-facing agent-type label to the ``agent_kind``
 *  discriminator value the backend expects. "OpenHands" is the
 *  friendly name for the standard LLM agent. */
const AGENT_KINDS = {
  openhands: "llm",
  acp: "acp",
} as const;

type AgentTypeKey = keyof typeof AGENT_KINDS;

/**
 * Settings page that picks between the OpenHands (LLM) agent and an
 * ACP agent. This page is deliberately *just the selector* — saving
 * here only writes ``agent_kind``. The ACP-specific configuration
 * (server / model / subprocess details) lives on the dedicated
 * ``ACP Server`` and ``ACP Model`` sidebar entries that appear when
 * ``agent_kind === "acp"``.
 *
 * Switching agent type also reshapes the rest of the sidebar:
 *
 * - ``agent_kind = "llm"`` → LLM, Condenser, Verification, MCP,
 *   Skills are all visible.
 * - ``agent_kind = "acp"`` → ACP Server and ACP Model appear;
 *   LLM-only sub-tabs (Condenser, Verification, MCP, Skills) hide.
 */
function AgentTypeSettingsScreen() {
  const { t } = useTranslation();
  const { data: settings } = useSettings();
  const { mutate: saveSettings, isPending } = useSaveSettings();

  const persistedAgentKind = (
    settings?.agent_settings as Record<string, unknown> | undefined
  )?.agent_kind;
  const persistedKey: AgentTypeKey =
    persistedAgentKind === "acp" ? "acp" : "openhands";

  const [selectedKey, setSelectedKey] =
    React.useState<AgentTypeKey>(persistedKey);
  React.useEffect(() => {
    setSelectedKey(persistedKey);
  }, [persistedKey]);

  const selectionDirty = selectedKey !== persistedKey;
  const isAcp = selectedKey === "acp";

  const agentTypeItems = React.useMemo(
    () => [
      {
        key: "openhands" satisfies AgentTypeKey,
        label: t(I18nKey.SETTINGS$AGENT_TYPE_OPENHANDS),
      },
      {
        key: "acp" satisfies AgentTypeKey,
        label: t(I18nKey.SETTINGS$AGENT_TYPE_ACP),
      },
    ],
    [t],
  );

  const handleSave = React.useCallback(() => {
    saveSettings(
      {
        agent_settings: { agent_kind: AGENT_KINDS[selectedKey] },
      },
      {
        onSuccess: () => displaySuccessToast(t(I18nKey.SETTINGS$SAVED_WARNING)),
        onError: (error: AxiosError) => {
          const msg = retrieveAxiosErrorMessage(error);
          displayErrorToast(msg || t(I18nKey.ERROR$GENERIC));
        },
      },
    );
  }, [saveSettings, selectedKey, t]);

  return (
    <div
      className="flex flex-col gap-6"
      data-testid="agent-type-settings-screen-wrapper"
    >
      <div className="flex flex-col gap-4 rounded-md border border-[color:var(--color-border,rgba(255,255,255,0.1))] p-4">
        <Typography.H3>{t(I18nKey.SETTINGS$AGENT_TYPE_TITLE)}</Typography.H3>
        <SettingsDropdownInput
          testId="agent-type-dropdown"
          name="agent-type"
          label={t(I18nKey.SETTINGS$AGENT_TYPE_LABEL)}
          items={agentTypeItems}
          selectedKey={selectedKey}
          isClearable={false}
          onSelectionChange={(key) => {
            if (key === "openhands" || key === "acp") {
              setSelectedKey(key);
            }
          }}
          wrapperClassName="w-full max-w-sm"
        />
        <p className="text-sm text-tertiary-alt">
          {isAcp
            ? t(I18nKey.SETTINGS$AGENT_TYPE_ACP_DESCRIPTION)
            : t(I18nKey.SETTINGS$AGENT_TYPE_OPENHANDS_DESCRIPTION)}
        </p>
      </div>

      <div className="sticky bottom-0 bg-base py-4">
        <BrandButton
          testId="save-button"
          type="button"
          variant="primary"
          isDisabled={isPending || !selectionDirty}
          onClick={handleSave}
        >
          {isPending
            ? t(I18nKey.SETTINGS$SAVING)
            : t(I18nKey.SETTINGS$SAVE_CHANGES)}
        </BrandButton>
      </div>
    </div>
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default AgentTypeSettingsScreen;
