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
import {
  getAcpPreferredDefaultModel,
  getAcpProvider,
  labelForAcpModel,
  resolveEffectiveAcpModel,
} from "#/constants/acp-providers";
import {
  useAcpModelChoices,
  type AcpModelChoice,
} from "#/hooks/use-acp-model-choices";
import {
  composeAcpModelId,
  getAcpEffortLevels,
  parseAcpModelId,
} from "#/utils/acp-model-id";
import { useSwitchAcpModel } from "#/hooks/mutation/use-switch-acp-model";

export interface ChatInputModelState {
  isAcpContext: boolean;
  displayModel: string | null;
  currentModelId: string | null;
  /**
   * `currentModelId`'s base model (see {@link parseAcpModelId}) — identical
   * to `currentModelId` unless it's a composite "<base>/<effort>" session
   * id, in which case this is the bare base. The picker uses this as a
   * fallback match so a composite id (e.g. "sonnet/high") still highlights
   * its bare "sonnet" row (M4/M5 own actually offering effort rows).
   */
  currentModelBaseId: string | null;
  availableAcpModels: AcpModelChoice[];
  showAcpPicker: boolean;
  switchConversationId: string | null;
  destinationPath: AcpModelContext["destinationPath"];
  destinationLabel: string;
  /**
   * The ACP server key backing `currentModelId`/`availableAcpModels` — the
   * `acpServer` {@link parseAcpModelId}/{@link composeAcpModelId} need to
   * know which effort suffixes are valid for the running provider. Exposed
   * so chat-input-model.tsx's `handleSelectAcpModel` can compose the current
   * effort onto a newly picked base model (M5 effort preservation) without
   * re-deriving the server key itself.
   */
  acpServerKey: string | null;
  /**
   * Effort level the effort UI should mark as current. Precedence:
   * the session's live `acp_current_effort` (threaded in M3, only meaningful
   * inside an active ACP conversation) → the effort parsed off a composite
   * `currentModelId` (e.g. "sonnet/high" → "high", via
   * {@link parseAcpModelId}) → the UI-only `"default"` sentinel
   * {@link composeAcpModelId} treats as "no suffix". Always a concrete
   * string so callers never need a null check to compare against a picker
   * row.
   */
  currentEffort: string;
  /**
   * Effort levels the effort UI should offer, or `null` to hide it entirely.
   * Prefers the session's live `acp_available_efforts` when the server
   * reports any (already includes the `"default"` sentinel — see
   * `AppConversation.acp_available_efforts`); falls back to the static
   * per-server list from {@link getAcpEffortLevels}. Gated identically to
   * `showAcpPicker` — an effort section should never render somewhere the
   * model list itself wouldn't (same ACP-context + cloud-permission gating).
   */
  availableEfforts: string[] | null;
  /**
   * Switch the live/default effort level, holding the current base model
   * fixed — the effort analog of picking a row in `availableAcpModels`. A
   * no-op when `effort` already equals `currentEffort` (or there's no
   * current base model to compose onto). Composes `currentModelBaseId` +
   * `effort` back into the raw ACP model id via {@link composeAcpModelId}
   * and routes through the same `useSwitchAcpModel` mutation a base-model
   * pick uses — live in-session when `switchConversationId` is set,
   * persisted to the active profile / legacy agent_settings on the home
   * page otherwise (identical dual-target behavior to a model pick, since
   * it's the same mutation).
   */
  handleSelectAcpEffort: (effort: string) => void;
}

export function useChatInputModelState(): ChatInputModelState {
  const { data: conversation } = useActiveConversation();
  const { data: settings } = useSettings();
  const { conversationId } = useOptionalConversationId();
  const { backend } = useActiveBackend();
  const canManageOrgProfiles = useCanManageOrgProfiles();
  const switchAcpModel = useSwitchAcpModel();
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
      providerDefault: getAcpPreferredDefaultModel(acpServerKey),
    });
  } else {
    currentModelId = conversation?.llm_model ?? settings?.llm_model ?? null;
  }

  const displayModel =
    currentModelId && isAcpContext
      ? (labelForAcpModel(acpServerKey, currentModelId) ?? currentModelId)
      : currentModelId;

  // The active AgentProfile's stable UUID, used to key the "remembered
  // custom model" store (see useAcpCustomModelsStore / useAcpModelChoices).
  // On the home page this is the active profile fetched above; in a
  // conversation it's the profile the conversation itself launched from.
  // A conversation started off legacy `agent_settings` (no profile) has
  // none — `undefined` there means "no custom entries offered/rememberable",
  // matching useAcpModelChoices' documented pre-M2 behavior for an unknown
  // profile id.
  const profileId = isActiveAcpConversation
    ? (conversation?.launched_agent_profile?.agent_profile_id ?? undefined)
    : isHomeAcp
      ? (activeAcpProfile?.id ?? undefined)
      : undefined;

  // Live models the ACP session itself currently reports — only meaningful
  // inside an active ACP conversation (the home page has no running
  // session); `undefined` there, and also on an agent-server too old to
  // surface ConversationInfo.available_models. See AppConversation.acp_live_models.
  const liveModels = isActiveAcpConversation
    ? conversation?.acp_live_models
    : undefined;

  // Live effort fields (agent-canvas M5) — same session-only gating as
  // liveModels above: undefined on the home page (no running session) and
  // for a non-ACP active conversation (the adapter already nulls both there,
  // but the explicit gate documents the intent regardless).
  const liveCurrentEffort = isActiveAcpConversation
    ? conversation?.acp_current_effort
    : undefined;
  const liveAvailableEfforts = isActiveAcpConversation
    ? conversation?.acp_available_efforts
    : undefined;

  const { choices: availableAcpModels } = useAcpModelChoices({
    acpServer: acpServerKey,
    curated: acpProvider?.available_models ?? [],
    profileId,
    liveModels,
    // Skip the models.dev catalog fetch outside an ACP context — this hook
    // still runs on every chat render (Rules of Hooks), including plain
    // OpenHands conversations that will never show the picker.
    enabled: isAcpContext,
  });

  const currentModelBaseId = currentModelId
    ? parseAcpModelId(currentModelId, acpServerKey).base
    : null;

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

  const currentEffort =
    liveCurrentEffort ??
    (currentModelId
      ? parseAcpModelId(currentModelId, acpServerKey).effort
      : null) ??
    "default";

  // Only expose effort switching wherever the model picker itself is shown
  // (same ACP-context + cloud-permission gating as showAcpPicker) — an
  // effort section with no model list to sit under wouldn't make sense.
  const availableEfforts = showAcpPicker
    ? liveAvailableEfforts && liveAvailableEfforts.length > 0
      ? liveAvailableEfforts
      : getAcpEffortLevels(acpServerKey)
    : null;

  const handleSelectAcpEffort = (effort: string) => {
    if (!currentModelBaseId || effort === currentEffort) return;
    switchAcpModel.mutate({
      conversationId: switchConversationId,
      model: composeAcpModelId(currentModelBaseId, effort, acpServerKey),
    });
  };

  return {
    isAcpContext,
    displayModel,
    currentModelId,
    currentModelBaseId,
    availableAcpModels,
    showAcpPicker,
    switchConversationId,
    destinationPath,
    destinationLabel,
    acpServerKey: acpServerKey ?? null,
    currentEffort,
    availableEfforts,
    handleSelectAcpEffort,
  };
}
