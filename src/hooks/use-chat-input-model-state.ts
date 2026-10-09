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
import { I18nKey } from "#/i18n/declaration";

export interface ChatInputModelState {
  isAcpContext: boolean;
  displayModel: string | null;
  currentModelId: string | null;
  availableAcpModels: ACPModelOption[];
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
  const discovered = useAcpModelDiscovery(acpProvider ? acpServerKey : null);
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
  if (isActiveAcpConversation) {
    // ACP conversations store llm_model as the acp_model (persisted at
    // creation time). Use it directly if available; fall back to the
    // settings-configured model or provider default so the chip stays visible.
    currentModelId =
      conversation?.llm_model ??
      resolveEffectiveAcpModel({
        configured: acpConfiguredModel,
        providerDefault: getAcpPreferredDefaultModel(acpServerKey),
      });
  } else if (isHomeAcp) {
    currentModelId = resolveEffectiveAcpModel({
      configured: acpConfiguredModel,
      // Preferred default (Vertex-safe for Gemini) — must match what the
      // start request would substitute for an unconfigured model.
      providerDefault:
        getAcpPreferredDefaultModel(acpServerKey) ?? discovered.defaultModelId,
    });
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
  const switchConversationId = isActiveAcpConversation
    ? (conversationId ?? null)
    : null;

  return {
    isAcpContext,
    displayModel,
    currentModelId,
    availableAcpModels,
    showAcpPicker,
    switchConversationId,
    destinationPath,
    destinationLabel,
  };
}
