import { QueryClient } from "@tanstack/react-query";
import { ConversationClient } from "@openhands/typescript-client/clients";
import type { StartGoalRequest } from "@openhands/typescript-client";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { pauseCloudSandbox } from "#/api/cloud/conversation-service.api";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { AgentKind } from "#/types/settings";

type ExecutionStatusValue = AppConversation["execution_status"];

export interface ResolveLaunchProfileOptions {
  /**
   * Explicitly requested agent profile id, or the backend's
   * `active_agent_profile_id` when the home launcher didn't name one (#3727).
   * Undefined when no agent profile is involved at all (plain agent_settings
   * launch).
   */
  requestedAgentProfileId?: string | null;
  /** Agent-profile list, as fetched through the shared query cache. */
  agentProfiles?: AgentProfileListResponse;
  /**
   * LLM-profile list; carries the account-wide `active_profile` that the
   * home-page LLM dropdown activates. Undefined when the list couldn't be
   * fetched (older backend, or the fetch failed).
   */
  llmProfiles?: ProfileListResponse;
  /**
   * Cloud backends never carry an `agent_settings` payload to fall back to, so
   * every profile-path downgrade below is local-only (see #3727 review).
   */
  isCloud: boolean;
}

/**
 * Why a launch that would have used an AgentProfile id instead fell back to
 * the agent_settings path. Kept for diagnosable traces; the UI treats all
 * downgrades as the plain launch path.
 */
export type LaunchProfileDowngradeReason =
  | "default-baseline"
  | "missing-llm-profile-ref"
  | "dropdown-llm-profile-selected";

export interface ResolveLaunchProfileResult {
  /** `agent_profile_id` to send; undefined → launch from agent_settings. */
  effectiveAgentProfileId?: string;
  /** Kind of the resolved agent profile, forwarded with the id (#3727). */
  agentProfileKind?: AgentKind;
  /**
   * The LLM profile the conversation will actually run, for the #1082
   * metadata stamp: the pinned `llm_profile_ref` when a named OpenHands
   * profile drives the launch, otherwise the account-wide active LLM profile
   * (the home-dropdown selection). Null when neither is known.
   */
  launchLlmProfileRef: string | null;
  downgradeReason?: LaunchProfileDowngradeReason;
}

/**
 * Resolve which AgentProfile (if any) a new conversation launches from, and
 * which LLM profile it will therefore run.
 *
 * The active AgentProfile is the default launch profile for new conversations
 * (#3727). A named OpenHands profile pins an LLM profile via
 * `llm_profile_ref`, and launching through it makes the server run that pinned
 * model. That pin is what the home-page LLM dropdown overrides: the dropdown
 * activates an account-wide LLM profile (`active_profile` on
 * `/api/profiles`), so when it differs from the pinned ref the user explicitly
 * picked a different model and the selection must win (#16539) — launch via
 * agent_settings instead, which reflects the active LLM profile.
 *
 * The seeded OpenHands `default` profile is the enriched baseline, not a
 * deliberate profile pick — it mirrors global agent_settings, so it always
 * launches via agent_settings so the canvas-only enrichments the
 * profile-resolution path drops survive for the common home-launch (the
 * `<RUNTIME_SERVICES>` system-message suffix and project-skill loading).
 * Named profiles are deliberate custom configs and still use the profile path
 * unless the dropdown overrides their model.
 *
 * ACP profiles carry no LLM profile at all, so they're never gated here and
 * always keep the profile path.
 */
export function resolveLaunchProfile(
  options: ResolveLaunchProfileOptions,
): ResolveLaunchProfileResult {
  const { requestedAgentProfileId, agentProfiles, llmProfiles, isCloud } =
    options;

  // No agent profile → the plain agent_settings path, which reflects the
  // account-wide active LLM profile (the home-dropdown selection).
  if (!requestedAgentProfileId) {
    return { launchLlmProfileRef: llmProfiles?.active_profile ?? null };
  }

  const resolvedAgentProfile = agentProfiles?.profiles?.find(
    (profile) => profile.id === requestedAgentProfileId,
  );

  // A requested profile the list doesn't know (stale pointer) still launches
  // by id; the server resolves or rejects it.
  if (!resolvedAgentProfile) {
    return {
      effectiveAgentProfileId: requestedAgentProfileId,
      launchLlmProfileRef: llmProfiles?.active_profile ?? null,
    };
  }

  const isOpenHands = resolvedAgentProfile.agent_kind === "openhands";

  // The seeded `default` baseline is global agent_settings, not a deliberate
  // profile pick: launch it via agent_settings so the canvas-only enrichments
  // survive. Scoped to OpenHands (an ACP `default` must keep the profile
  // path) and to local (cloud never writes agent_settings, so it always
  // resolves `default` server-side via agent_profile_id).
  if (
    !isCloud &&
    isOpenHands &&
    resolvedAgentProfile.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME
  ) {
    return {
      effectiveAgentProfileId: undefined,
      agentProfileKind: resolvedAgentProfile.agent_kind,
      launchLlmProfileRef: llmProfiles?.active_profile ?? null,
      downgradeReason: "default-baseline",
    };
  }

  // Named OpenHands profiles pin an LLM profile by ref. Validate the ref
  // exists: the agent-server seeds a `default` profile whose ref can point at
  // an LLM profile that doesn't exist (fresh store, or one configured with
  // named profiles only); launching from it 404s and would brick home-launch.
  // When the ref is valid it normally drives the launch — unless the user
  // explicitly selected a different model in the home LLM dropdown, in which
  // case the account-wide active LLM profile wins over the pinned ref (#16539).
  if (isOpenHands && resolvedAgentProfile.llm_profile_ref) {
    const llmProfileExists =
      llmProfiles?.profiles.some(
        (profile) => profile.name === resolvedAgentProfile.llm_profile_ref,
      ) ?? false;

    if (!llmProfileExists) {
      // Launching from a profile whose ref doesn't resolve 404s ("LLM
      // profile '<ref>' not found") and would brick home-launch. The
      // agent-server seeds a `default` profile whose llm_profile_ref can
      // point at an LLM profile that doesn't exist (fresh store, or one
      // configured with named profiles only), and any named profile can
      // dangle the same way. agent_settings reflects the active LLM, so the
      // fallback degrades cleanly until the seed mirrors it (SDK #3933).
      return {
        effectiveAgentProfileId: undefined,
        agentProfileKind: resolvedAgentProfile.agent_kind,
        launchLlmProfileRef: llmProfiles?.active_profile ?? null,
        downgradeReason: "missing-llm-profile-ref",
      };
    }

    if (
      !isCloud &&
      llmProfiles?.active_profile &&
      llmProfiles.active_profile !== resolvedAgentProfile.llm_profile_ref
    ) {
      // The home LLM dropdown selected a different profile than the one the
      // active named AgentProfile pins: honor the dropdown by launching from
      // agent_settings (which reflects the active LLM profile) instead of the
      // pinned ref, so the next conversation runs the model the user picked.
      // Local-only: cloud has no agent_settings fallback payload.
      return {
        effectiveAgentProfileId: undefined,
        agentProfileKind: resolvedAgentProfile.agent_kind,
        launchLlmProfileRef: llmProfiles.active_profile,
        downgradeReason: "dropdown-llm-profile-selected",
      };
    }

    return {
      effectiveAgentProfileId: requestedAgentProfileId,
      agentProfileKind: resolvedAgentProfile.agent_kind,
      launchLlmProfileRef: resolvedAgentProfile.llm_profile_ref,
    };
  }

  // An OpenHands profile without a ref (shouldn't occur — the openhands
  // variant requires llm_profile_ref) or an ACP profile: keep the profile
  // path; the LLM comes from the active profile.
  return {
    effectiveAgentProfileId: requestedAgentProfileId,
    agentProfileKind: resolvedAgentProfile.agent_kind,
    launchLlmProfileRef: llmProfiles?.active_profile ?? null,
  };
}

const fetchConversationData = async (
  conversationId: string,
): Promise<{
  conversationUrl: string | null;
  sessionApiKey: string | null;
  sandboxId: string | null;
}> => {
  const conversations =
    await AgentServerConversationService.batchGetAppConversations([
      conversationId,
    ]);

  const appConversation = conversations[0];
  if (!appConversation) {
    throw new Error(`V1 conversation not found: ${conversationId}`);
  }

  return {
    conversationUrl: appConversation.conversation_url,
    sessionApiKey: appConversation.session_api_key,
    sandboxId: appConversation.sandbox_id,
  };
};

/**
 * Stop a running conversation.
 * - Cloud mode: Pauses the sandbox (waits for current LLM call to finish).
 * - Local mode: Interrupts immediately (cancels in-flight requests).
 */
export const pauseConversation = async (conversationId: string) => {
  const { conversationUrl, sessionApiKey, sandboxId } =
    await fetchConversationData(conversationId);

  if (getActiveBackend().backend.kind === "cloud") {
    if (!sandboxId) {
      throw new Error(
        `Cannot stop runtime: cloud conversation ${conversationId} has no sandbox_id.`,
      );
    }
    await pauseCloudSandbox(sandboxId);
    return { success: true };
  }

  // In local mode, use /interrupt instead of /pause so in-flight LLM
  // requests are cancelled immediately rather than waiting for the
  // current call to finish.
  return new ConversationClient(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).interruptConversation(conversationId);
};

/**
 * Ask the agent a side question on a V1 conversation
 */
export const askAgent = async (
  conversationId: string,
  question: string,
): Promise<{ response: string }> => {
  const { conversationUrl, sessionApiKey } =
    await fetchConversationData(conversationId);
  return new ConversationClient(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).askAgent(conversationId, question);
};

/**
 * Start a `/goal` loop on a V1 conversation. The agent server drives the agent
 * toward the objective, judging completion after each run until it is done or
 * `max_iterations` is reached, streaming progress as goal
 * ConversationStateUpdateEvents over the conversation's event stream.
 */
export const startGoal = async (
  conversationId: string,
  request: StartGoalRequest,
): Promise<void> => {
  const { conversationUrl, sessionApiKey } =
    await fetchConversationData(conversationId);
  await new ConversationClient(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).startGoal(conversationId, request);
};

/**
 * Stop the active `/goal` loop. The backend only cancels the background loop
 * (recording an `interrupted` status so {@link resumeGoal} can continue it) and
 * deliberately leaves the in-flight agent turn running, so callers should also
 * interrupt the conversation (e.g. `pauseConversation`) to actually halt it.
 */
export const stopGoal = async (conversationId: string): Promise<void> => {
  const { conversationUrl, sessionApiKey } =
    await fetchConversationData(conversationId);
  await new ConversationClient(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).stopGoal(conversationId);
};

/** Resume the last interrupted `/goal` loop in this conversation. */
export const resumeGoal = async (conversationId: string): Promise<void> => {
  const { conversationUrl, sessionApiKey } =
    await fetchConversationData(conversationId);
  await new ConversationClient(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).resumeGoal(conversationId);
};

export const resumeConversation = async (conversationId: string) => {
  const { conversationUrl, sessionApiKey } =
    await fetchConversationData(conversationId);
  return new ConversationClient(
    getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
  ).runConversation(conversationId);
};

/**
 * Patch arbitrary fields on a cached AppConversation in both the single-item
 * and paginated list query caches.  Prefer this over the narrower
 * `updateConversationExecutionStatusInCache` when you need to update more than
 * one field atomically (e.g. `execution_status` + `sandbox_status` together).
 */
export const patchConversationInCache = (
  queryClient: QueryClient,
  conversationId: string,
  patch: Partial<AppConversation>,
): void => {
  // useUserConversation stores data under a 5-part key that includes the active
  // backend id and org id. Use setQueriesData with prefix matching so the
  // update reaches whichever (backend, org) variant is currently mounted.
  queryClient.setQueriesData<AppConversation | null>(
    { queryKey: ["user", "conversation", conversationId] },
    (oldData) => (oldData ? { ...oldData, ...patch } : oldData),
  );

  queryClient.setQueriesData<{
    pages: Array<{ items: AppConversation[] }>;
  }>({ queryKey: ["user", "conversations"] }, (oldData) => {
    if (!oldData) return oldData;
    return {
      ...oldData,
      pages: oldData.pages.map((page) => ({
        ...page,
        items: page.items.map((conv) =>
          conv.id === conversationId ? { ...conv, ...patch } : conv,
        ),
      })),
    };
  });
};

export const updateConversationExecutionStatusInCache = (
  queryClient: QueryClient,
  conversationId: string,
  execution_status: ExecutionStatusValue,
): void =>
  patchConversationInCache(queryClient, conversationId, { execution_status });

export const updateConversationLlmModelInCache = (
  queryClient: QueryClient,
  conversationId: string,
  llm_model: string,
): void => patchConversationInCache(queryClient, conversationId, { llm_model });

export const invalidateConversationQueries = (
  queryClient: QueryClient,
  conversationId: string,
): void => {
  queryClient.invalidateQueries({
    queryKey: ["user", "conversation", conversationId],
  });
  queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
  queryClient.invalidateQueries({
    queryKey: ["v1-batch-get-app-conversations"],
  });
  queryClient.invalidateQueries({ queryKey: ["unified", "vscode_url"] });
};
