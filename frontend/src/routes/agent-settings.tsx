import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { useSettings } from "#/hooks/query/use-settings";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export const handle = { hideTitle: true };

type AgentType = "openhands" | "claude-code" | "codex" | "gemini-cli";
type TabType = "basic" | "advanced";

interface AgentOption {
  key: AgentType;
  label: string;
}

const AGENT_OPTIONS: AgentOption[] = [
  { key: "openhands", label: "OpenHands" },
  { key: "claude-code", label: "Claude Code" },
  { key: "codex", label: "Codex" },
  { key: "gemini-cli", label: "Gemini CLI" },
];

const API_KEY_LABELS: Partial<Record<AgentType, string>> = {
  "claude-code": "Anthropic API Key",
  codex: "OpenAI API Key",
  "gemini-cli": "Google API Key",
};

const DEFAULT_COMMANDS: Partial<Record<AgentType, string>> = {
  "claude-code": "npx -y @agentclientprotocol/claude-agent-acp",
  codex: "npx -y @zed-industries/codex-acp",
  "gemini-cli": "npx -y @google/gemini-cli --acp",
};

function AgentSettingsScreen() {
  const { t } = useTranslation();
  const { data: settings, isLoading } = useSettings();
  const { mutate: saveSettings, isPending: isSaving } = useSaveSettings();

  const [tab, setTab] = useState<TabType>("basic");
  const [agentType, setAgentType] = useState<AgentType>("openhands");
  const [apiKey, setApiKey] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [envJson, setEnvJson] = useState("{}");
  const [envError, setEnvError] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  useEffect(() => {
    if (!settings) return;
    const kind = settings.agent_settings?.kind;
    if (kind === "acp") {
      const server =
        (settings.agent_settings?.acp_server as string) ?? "claude-code";
      setAgentType(server as AgentType);
      const acpCommand = settings.agent_settings?.acp_command as
        | string[]
        | undefined;
      setCommand(acpCommand ? acpCommand.join(" ") : "");
      const acpArgs = settings.agent_settings?.acp_args as string[] | undefined;
      setArgs(acpArgs ? acpArgs.join(" ") : "");
      const acpEnv = settings.agent_settings?.acp_env as
        | Record<string, string>
        | undefined;
      setEnvJson(
        acpEnv && Object.keys(acpEnv).length > 0
          ? JSON.stringify(acpEnv, null, 2)
          : "{}",
      );
    } else {
      setAgentType("openhands");
    }
    setIsDirty(false);
  }, [settings]);

  const handleAgentTypeChange = (key: React.Key | null) => {
    if (!key) return;
    setAgentType(key as AgentType);
    setApiKey("");
    setIsDirty(true);
  };

  const handleSave = () => {
    if (envError) return;

    let parsedEnv: Record<string, string> = {};
    if (agentType !== "openhands" && envJson.trim()) {
      try {
        parsedEnv = JSON.parse(envJson);
      } catch {
        setEnvError(t(I18nKey.SETTINGS$MCP_ERROR_INVALID_JSON));
        return;
      }
    }

    let agentSettingsDiff: Record<string, unknown>;

    if (agentType === "openhands") {
      agentSettingsDiff = { kind: "llm" };
    } else {
      const commandParts = command.trim()
        ? command.trim().split(/\s+/)
        : (DEFAULT_COMMANDS[agentType] ?? "").split(/\s+/);

      const argParts = args.trim() ? args.trim().split(/\s+/) : [];

      agentSettingsDiff = {
        kind: "acp",
        acp_server: agentType,
        acp_command: commandParts,
        acp_args: argParts,
        acp_env: parsedEnv,
        ...(apiKey.trim() ? { llm: { api_key: apiKey.trim() } } : {}),
      };
    }

    saveSettings(
      { agent_settings_diff: agentSettingsDiff },
      {
        onError: (error) => {
          const message = retrieveAxiosErrorMessage(error as AxiosError);
          displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
        },
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$AGENT_SAVED));
          setIsDirty(false);
          setApiKey("");
        },
      },
    );
  };

  const isAcp = agentType !== "openhands";
  const apiKeyLabel = API_KEY_LABELS[agentType];
  const apiKeyIsSet =
    isAcp &&
    settings?.llm_api_key_set &&
    settings?.agent_settings?.kind === "acp";

  const tabButtonClass = (active: boolean) =>
    `px-4 py-2 text-sm font-medium rounded-t transition-colors ${
      active
        ? "bg-[#1f1f1f] text-white border-b-2 border-white"
        : "text-[#8C8C8C] hover:text-white"
    }`;

  if (isLoading) return null;

  return (
    <div className="flex flex-col gap-6 pb-8 max-w-2xl">
      <div>
        <Typography.H2 className="mb-2">
          {t(I18nKey.SETTINGS$AGENT_PAGE_TITLE)}
        </Typography.H2>
        <Typography.Paragraph className="text-sm text-[#A3A3A3]">
          {t(I18nKey.SETTINGS$AGENT_PAGE_DESCRIPTION)}
        </Typography.Paragraph>
      </div>

      <div className="flex border-b border-[#333]">
        <button
          type="button"
          className={tabButtonClass(tab === "basic")}
          onClick={() => setTab("basic")}
        >
          {t(I18nKey.SETTINGS$AGENT_BASIC_TAB)}
        </button>
        {isAcp && (
          <button
            type="button"
            className={tabButtonClass(tab === "advanced")}
            onClick={() => setTab("advanced")}
          >
            {t(I18nKey.SETTINGS$AGENT_ADVANCED_TAB)}
          </button>
        )}
      </div>

      {tab === "basic" && (
        <div className="flex flex-col gap-6">
          <SettingsDropdownInput
            testId="agent-type-selector"
            name="agent-type"
            label={t(I18nKey.SETTINGS$AGENT_TYPE_LABEL)}
            items={AGENT_OPTIONS.map((o) => ({ key: o.key, label: o.label }))}
            selectedKey={agentType}
            onSelectionChange={handleAgentTypeChange}
          />

          {isAcp && apiKeyLabel && (
            <SettingsInput
              testId="agent-api-key-input"
              label={apiKeyLabel}
              type="password"
              className="w-full"
              value={apiKey}
              placeholder={apiKeyIsSet ? "<hidden>" : ""}
              onChange={(value) => {
                setApiKey(value);
                setIsDirty(true);
              }}
              startContent={apiKeyIsSet ? <KeyStatusIcon isSet /> : undefined}
            />
          )}
        </div>
      )}

      {tab === "advanced" && isAcp && (
        <div className="flex flex-col gap-6">
          <SettingsInput
            testId="agent-command-input"
            label={t(I18nKey.SETTINGS$AGENT_ADVANCED_COMMAND)}
            type="text"
            className="w-full"
            value={command}
            placeholder={DEFAULT_COMMANDS[agentType] ?? ""}
            onChange={(value) => {
              setCommand(value);
              setIsDirty(true);
            }}
          />

          <SettingsInput
            testId="agent-args-input"
            label={t(I18nKey.SETTINGS$AGENT_ADVANCED_ARGS)}
            type="text"
            className="w-full"
            value={args}
            placeholder={t(I18nKey.SETTINGS$AGENT_ADVANCED_ARGS_PLACEHOLDER)}
            onChange={(value) => {
              setArgs(value);
              setIsDirty(true);
            }}
          />

          <div className="flex flex-col gap-2.5">
            <span className="text-sm">
              {t(I18nKey.SETTINGS$AGENT_ADVANCED_ENV)}
            </span>
            <textarea
              data-testid="agent-env-input"
              className="bg-tertiary border border-[#717888] rounded-sm p-2 text-sm font-mono text-white placeholder:italic placeholder:text-[#717888] min-h-[100px] resize-y focus:outline-none focus:border-white"
              value={envJson}
              placeholder={t(I18nKey.SETTINGS$AGENT_ADVANCED_ENV_PLACEHOLDER)}
              onChange={(e) => {
                setEnvJson(e.target.value);
                setEnvError(null);
                setIsDirty(true);
                try {
                  JSON.parse(e.target.value);
                } catch {
                  setEnvError(t(I18nKey.SETTINGS$MCP_ERROR_INVALID_JSON));
                }
              }}
            />
            {envError && (
              <span className="text-xs text-red-400">{envError}</span>
            )}
          </div>
        </div>
      )}

      <div className="flex gap-3">
        <BrandButton
          testId="agent-save-button"
          type="button"
          variant="primary"
          isDisabled={isSaving || !isDirty || !!envError}
          onClick={handleSave}
        >
          {isSaving
            ? t(I18nKey.SETTINGS$AGENT_SAVING)
            : t(I18nKey.SETTINGS$AGENT_SAVE)}
        </BrandButton>
      </div>
    </div>
  );
}

export default AgentSettingsScreen;
