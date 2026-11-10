import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSettings } from "#/hooks/query/use-settings";
import {
  type AcpModelContext,
  useAcpModelContext,
} from "#/hooks/use-acp-model-context";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import { useActiveAcpProfileDetail } from "#/hooks/query/use-active-acp-profile-detail";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useAcpModelDiscovery } from "#/hooks/query/use-acp-model-discovery";
import { useAcpSessionModels } from "#/hooks/query/use-acp-session-models";
import {
  getAcpPreferredDefaultModel,
  getAcpProvider,
  labelForAcpModel,
  resolveEffectiveAcpModel,
  type ACPModelOption,
} from "#/constants/acp-providers";
import { parseCommand } from "#/utils/acp-command";
import { I18nKey } from "#/i18n/declaration";

export interface ChatInputModelState {
  isAcpContext: boolean;
  displayModel: string | null;
  currentModelId: string | null;
  /** The saved choice the picker marks; null is the agent's own default. */
  selectedModelId: string | null;
  availableAcpModels: ACPModelOption[];
  /** The picker's Agent default row, or null when it isn't offered. */
  agentDefaultLabel: string | null;
  showAcpPicker: boolean;
  switchConversationId: string | null;
  destinationPath: AcpModelContext["destinationPath"];
  destinationLabel: string;
}

export function useChatInputModelState(): ChatInputModelState {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { data: settings } = useSettings();
  const { conversationId } = useOptionalConversationId();
  const { backend } = useActiveBackend();
  const canManageOrgProfiles = useCanManageOrgProfiles();
  // The active ACP AgentProfile's own fields are the conversation launch
  // source (activation never writes agent_settings, so the global settings
  // may describe a different provider). Null in a conversation, while
  // loading, when the active profile isn't ACP, or on legacy backends
  // without the profiles surface — settings are the fallback in that window.
  const activeAcpProfile = useActiveAcpProfileDetail();
  const {
    isActiveAcpConversation,
    isHomeAcp,
    isAcpContext,
    destinationPath,
    destinationLabel,
  } = useAcpModelContext();

  const settingsAcpServerKey =
    typeof settings?.agent_settings?.acp_server === "string"
      ? settings.agent_settings.acp_server
      : null;
  const acpServerKey = isActiveAcpConversation
    ? conversation?.acp_server
    : isHomeAcp
      ? (activeAcpProfile?.acp_server ?? settingsAcpServerKey)
      : null;
  const acpProvider = isAcpContext ? getAcpProvider(acpServerKey) : undefined;
  const isCustomHomeAcp = isHomeAcp && acpServerKey === "custom";

  const effectiveAcpCommand = useMemo(() => {
    if (!isCustomHomeAcp) return null;
    const rawCommand =
      activeAcpProfile?.acp_command ??
      (typeof settings?.agent_settings?.acp_command === "string"
        ? settings.agent_settings.acp_command
        : Array.isArray(settings?.agent_settings?.acp_command)
          ? settings.agent_settings.acp_command
          : null);
    if (!rawCommand) return null;
    if (Array.isArray(rawCommand)) {
      return rawCommand.filter((c): c is string => typeof c === "string");
    }
    return parseCommand(rawCommand);
  }, [
    isCustomHomeAcp,
    activeAcpProfile?.acp_command,
    settings?.agent_settings?.acp_command,
  ]);

  const effectiveAcpArgs = useMemo(() => {
    if (!isCustomHomeAcp) return null;
    const rawArgs =
      activeAcpProfile?.acp_args ?? settings?.agent_settings?.acp_args;
    if (Array.isArray(rawArgs)) {
      return rawArgs.filter((a): a is string => typeof a === "string");
    }
    return null;
  }, [
    isCustomHomeAcp,
    activeAcpProfile?.acp_args,
    settings?.agent_settings?.acp_args,
  ]);

  const discoveryProvider = acpProvider
    ? acpServerKey
    : isCustomHomeAcp
      ? "custom"
      : null;

  const discoveryOptions = useMemo(
    () => ({
      enabled: isActiveAcpConversation || !!activeAcpProfile || isCustomHomeAcp,
      secretRefs:
        (isActiveAcpConversation
          ? conversation?.launched_agent_profile?.secret_refs
          : activeAcpProfile?.secret_refs) ?? null,
      ...(effectiveAcpCommand ? { acpCommand: effectiveAcpCommand } : {}),
      ...(effectiveAcpArgs ? { acpArgs: effectiveAcpArgs } : {}),
    }),
    [
      isActiveAcpConversation,
      activeAcpProfile,
      isCustomHomeAcp,
      conversation?.launched_agent_profile?.secret_refs,
      effectiveAcpCommand,
      effectiveAcpArgs,
    ],
  );

  // The agent's models depend on the secrets it may use: the ones the
  // conversation launched with, or the next launch's from the active profile.
  const discovered = useAcpModelDiscovery(discoveryProvider, discoveryOptions);
  // The session's own list arrives only after it starts.
  const sessionModels = useAcpSessionModels(
    isActiveAcpConversation ? conversation : null,
  );
  const liveModels = sessionModels.length ? sessionModels : discovered.models;

  const settingsAcpModel =
    typeof settings?.agent_settings?.acp_model === "string"
      ? settings.agent_settings.acp_model
      : null;
  // Home: read the model from the same source as the server key — mixing the
  // profile's provider with a stale settings model could pair e.g. a codex
  // provider with a claude model.
  const acpConfiguredModel =
    isHomeAcp && activeAcpProfile
      ? activeAcpProfile.acp_model
      : settingsAcpModel;

  let currentModelId: string | null = null;
  let selectedModelId: string | null = null;
  if (isActiveAcpConversation) {
    // The settings may name another provider's model, so never fall back to them.
    currentModelId =
      conversation?.llm_model ?? getAcpPreferredDefaultModel(acpServerKey);
    selectedModelId = currentModelId;
  } else if (isHomeAcp) {
    selectedModelId = resolveEffectiveAcpModel({
      configured: acpConfiguredModel,
      // Preferred default (Vertex-safe for Gemini) — must match what the
      // start request would substitute for an unconfigured model.
      providerDefault: getAcpPreferredDefaultModel(acpServerKey),
    });
    currentModelId = selectedModelId ?? discovered.defaultModelId;
  } else {
    currentModelId = conversation?.llm_model ?? settings?.llm_model ?? null;
  }

  let displayModel = currentModelId;
  if (isAcpContext) {
    displayModel =
      labelForAcpModel(currentModelId, liveModels) ??
      // The next conversation starts on a default no one has reported yet.
      (isHomeAcp ? t(I18nKey.SETTINGS$AGENT_MODEL_AGENT_DEFAULT) : null);
  }
  const availableAcpModels = isAcpContext ? liveModels : [];
  // A home-page pick persists into the active ACP profile, which on cloud is
  // org-owned — hide the selectable rows from members who'd only get a 403.
  // Conversation-scoped switches (blank or started) stay member-allowed.
  const canPersistHomeAcpModel =
    !isHomeAcp || backend.kind !== "cloud" || canManageOrgProfiles;
  const showAcpPicker =
    isAcpContext && availableAcpModels.length > 0 && canPersistHomeAcpModel;
  let agentDefaultLabel: string | null = null;
  if (
    isHomeAcp &&
    (acpProvider || isCustomHomeAcp) &&
    !getAcpPreferredDefaultModel(acpServerKey)
  ) {
    const defaultModelLabel = labelForAcpModel(
      discovered.defaultModelId,
      liveModels,
    );
    agentDefaultLabel = defaultModelLabel
      ? t(I18nKey.SETTINGS$AGENT_MODEL_AGENT_DEFAULT_IS, {
          model: defaultModelLabel,
        })
      : t(I18nKey.SETTINGS$AGENT_MODEL_AGENT_DEFAULT);
  }
  const switchConversationId = isActiveAcpConversation
    ? (conversationId ?? null)
    : null;

  return {
    isAcpContext,
    displayModel,
    currentModelId,
    selectedModelId,
    availableAcpModels,
    agentDefaultLabel,
    showAcpPicker,
    switchConversationId,
    destinationPath,
    destinationLabel,
  };
}
