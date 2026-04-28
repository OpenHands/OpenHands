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
import {
  AcpServerKind,
  ACP_SERVER_DISPLAY_NAMES,
  ACP_API_KEY_LABELS,
  ACP_DEFAULT_COMMANDS,
  isAcpServerKind,
} from "#/constants/acp-agents";

export const handle = { hideTitle: true };

type AgentType = "openhands" | AcpServerKind;
type TabType = "basic" | "advanced";

interface AgentOption {
  key: AgentType;
  label: string;
}

const AGENT_OPTIONS: AgentOption[] = [
  { key: "openhands", label: "OpenHands" },
  ...Object.entries(ACP_SERVER_DISPLAY_NAMES).map(([key, label]) => ({
    key: key as AcpServerKind,
    label,
  })),
];

const API_KEY_LABELS = ACP_API_KEY_LABELS;
const DEFAULT_COMMANDS = ACP_DEFAULT_COMMANDS;

function AgentSettingsScreen() {
  const { t } = useTranslation();
  const { data: settings, isLoading } = useSettings();
  const { mutate: saveSettings, isPending: isSaving } = useSaveSettings();

  const [tab, setTab] = useState<TabType>("basic");
  const [agentType, setAgentType] = useState<AgentType>("openhands");
  const [apiKey, setApiKey] = useState("");
  // command and args are stored as arrays; the textarea shows one token per line
  const [command, setCommand] = useState<string[]>([]);
  const [args, setArgs] = useState<string[]>([]);
  const [envJson, setEnvJson] = useState("{}");
  const [envError, setEnvError] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  useEffect(() => {
    if (!settings) return;
    const kind = settings.agent_settings?.kind;
    if (kind === "acp") {
      const rawServer = settings.agent_settings?.acp_server;
      setAgentType(isAcpServerKind(rawServer) ? rawServer : "claude-code");

      const acpCommand = settings.agent_settings?.acp_command;
      setCommand(
        Array.isArray(acpCommand)
          ? acpCommand.filter((v): v is string => typeof v === "string")
          : [],
      );

      const acpArgs = settings.agent_settings?.acp_args;
      setArgs(
        Array.isArray(acpArgs)
          ? acpArgs.filter((v): v is string => typeof v === "string")
          : [],
      );

      const acpEnv = settings.agent_settings?.acp_env;
      const envObj =
        acpEnv != null && typeof acpEnv === "object" && !Array.isArray(acpEnv)
          ? (acpEnv as Record<string, string>)
          : {};
      setEnvJson(
        Object.keys(envObj).length > 0 ? JSON.stringify(envObj, null, 2) : "{}",
      );
    } else {
      setAgentType("openhands");
    }
    setIsDirty(false);
  }, [settings]);

  const handleAgentTypeChange = (key: React.Key | null) => {
    if (!key) return;
    const newType = key as AgentType;
    // Clear the API key when switching between different ACP providers to
    // avoid showing an Anthropic key in the OpenAI key field, etc.
    if (
      newType !== agentType &&
      isAcpServerKind(newType) &&
      isAcpServerKind(agentType)
    ) {
      setApiKey("");
    }
    // Advanced tab doesn't apply to OpenHands; reset to basic to avoid showing
    // a blank/stale advanced view.
    if (newType === "openhands" && tab === "advanced") {
      setTab("basic");
    }
    setAgentType(newType);
    setIsDirty(true);
  };

  const handleSave = () => {
    if (envError) return;

    let parsedEnv: Record<string, string> = {};
    if (agentType !== "openhands" && envJson.trim()) {
      try {
        const parsed: unknown = JSON.parse(envJson);
        if (
          typeof parsed !== "object" ||
          Array.isArray(parsed) ||
          parsed === null
        ) {
          setEnvError(t(I18nKey.SETTINGS$AGENT_ENV_MUST_BE_OBJECT));
          return;
        }
        if (
          !Object.values(parsed as Record<string, unknown>).every(
            (v) => typeof v === "string",
          )
        ) {
          setEnvError(t(I18nKey.SETTINGS$AGENT_ENV_VALUES_MUST_BE_STRINGS));
          return;
        }
        parsedEnv = parsed as Record<string, string>;
      } catch {
        setEnvError(t(I18nKey.SETTINGS$MCP_ERROR_INVALID_JSON));
        return;
      }
    }

    let agentSettingsDiff: Record<string, unknown>;

    if (agentType === "openhands") {
      agentSettingsDiff = { kind: "llm" };
    } else {
      const effectiveCommand =
        command.length > 0 ? command : (DEFAULT_COMMANDS[agentType] ?? []);
      agentSettingsDiff = {
        kind: "acp",
        acp_server: agentType,
        acp_command: effectiveCommand,
        acp_args: args,
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
  const apiKeyLabel = isAcp
    ? API_KEY_LABELS[agentType as AcpServerKind]
    : undefined;
  const apiKeyIsSet =
    isAcp &&
    settings?.llm_api_key_set &&
    settings?.agent_settings?.kind === "acp";

  const defaultCommandPlaceholder = isAcpServerKind(agentType)
    ? (DEFAULT_COMMANDS[agentType] ?? []).join("\n")
    : "";

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
          <div className="flex flex-col gap-2.5">
            <span className="text-sm">
              {t(I18nKey.SETTINGS$AGENT_ADVANCED_COMMAND)}
            </span>
            <textarea
              data-testid="agent-command-input"
              className="bg-tertiary border border-[#717888] rounded-sm p-2 text-sm font-mono text-white placeholder:italic placeholder:text-[#717888] min-h-[80px] resize-y focus:outline-none focus:border-white"
              value={command.join("\n")}
              placeholder={defaultCommandPlaceholder}
              onChange={(e) => {
                setCommand(
                  e.target.value
                    .split("\n")
                    .map((s) => s.trim())
                    .filter(Boolean),
                );
                setIsDirty(true);
              }}
            />
            <span className="text-xs text-[#717888]">
              {t(I18nKey.SETTINGS$AGENT_ADVANCED_COMMAND_HINT)}
            </span>
          </div>

          <div className="flex flex-col gap-2.5">
            <span className="text-sm">
              {t(I18nKey.SETTINGS$AGENT_ADVANCED_ARGS)}
            </span>
            <textarea
              data-testid="agent-args-input"
              className="bg-tertiary border border-[#717888] rounded-sm p-2 text-sm font-mono text-white placeholder:italic placeholder:text-[#717888] min-h-[80px] resize-y focus:outline-none focus:border-white"
              value={args.join("\n")}
              placeholder={t(I18nKey.SETTINGS$AGENT_ADVANCED_ARGS_PLACEHOLDER)}
              onChange={(e) => {
                setArgs(
                  e.target.value
                    .split("\n")
                    .map((s) => s.trim())
                    .filter(Boolean),
                );
                setIsDirty(true);
              }}
            />
            <span className="text-xs text-[#717888]">
              {t(I18nKey.SETTINGS$AGENT_ADVANCED_ARGS_HINT)}
            </span>
          </div>

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
                  const parsed: unknown = JSON.parse(e.target.value);
                  if (
                    typeof parsed !== "object" ||
                    Array.isArray(parsed) ||
                    parsed === null
                  ) {
                    setEnvError(t(I18nKey.SETTINGS$AGENT_ENV_MUST_BE_OBJECT));
                  } else if (
                    !Object.values(parsed as Record<string, unknown>).every(
                      (v) => typeof v === "string",
                    )
                  ) {
                    setEnvError(
                      t(I18nKey.SETTINGS$AGENT_ENV_VALUES_MUST_BE_STRINGS),
                    );
                  }
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
