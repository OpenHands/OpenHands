import type { QueryClient } from "@tanstack/react-query";
import type { AgentKind } from "@openhands/typescript-client";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { backendSupportsLaunchLlmOverride } from "#/api/agent-server-adapter";
import AgentServerConversationService, {
  type CreateConversationOptions,
} from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { mergeStoredConversationMetadata } from "#/api/conversation-metadata-store";
import {
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
  LLM_PROFILES_QUERY_KEYS,
} from "#/hooks/query/query-keys";

export interface ConversationLaunch {
  /** Unset when the launch sends the global `agent_settings` instead. */
  agentProfileId?: string;
  agentProfileKind?: AgentKind;
  /** Runs the agent profile with this LLM profile instead of its own. */
  llmProfileOverride?: string;
  /** The LLM profile the launch runs, when Canvas can tell. */
  llmProfile?: string;
  // Profile summaries omit these; detail fetch fills them for Claude skill overlay (#16905).
  agentProfileAcpServer?: string | null;
  agentProfileAcpCommand?: string | readonly string[] | null;
}

export interface ResolveConversationLaunchOptions {
  queryClient: QueryClient;
  backendId: string;
  orgId: string | null | undefined;
  isCloud: boolean;
  /** An explicit profile pick; omitted launches the active profile. */
  agentProfileId?: string;
  /** The LLM profile to run, such as a parent conversation's current model. */
  llmProfile?: string | null;
}

/** Decide which agent source and LLM a new conversation launches with. */
export async function resolveConversationLaunch({
  queryClient,
  backendId,
  orgId,
  isCloud,
  agentProfileId,
  llmProfile,
}: ResolveConversationLaunchOptions): Promise<ConversationLaunch> {
  // Await the list from the shared query cache: a send fired before the home
  // query resolves must still launch from the active profile. Do not fall back
  // to the global agent_settings when profile discovery fails: activation is
  // pointer-only, so those settings may describe a different agent.
  const agentProfiles: AgentProfileListResponse =
    await queryClient.ensureQueryData({
      queryKey: [...AGENT_PROFILES_QUERY_KEYS.all, backendId, orgId],
      queryFn: AgentProfilesService.listProfiles,
      ...AGENT_PROFILES_RETRY_OPTIONS,
      // A failure here rejects the launch, which reports it.
      meta: { disableToast: true },
    });

  // Await the list rather than a render snapshot: a send fired before the
  // LLM-profile query loads must still validate the ref.
  const readLlmProfiles = () =>
    queryClient
      .ensureQueryData({
        queryKey: [...LLM_PROFILES_QUERY_KEYS.all, backendId, orgId],
        queryFn: ProfilesService.listProfiles,
        retry: false,
        meta: { disableToast: true },
      })
      .catch(() => null);

  const requestedAgentProfileId =
    agentProfileId ?? agentProfiles.active_agent_profile_id ?? undefined;
  const profile = requestedAgentProfileId
    ? agentProfiles.profiles?.find((p) => p.id === requestedAgentProfileId)
    : undefined;

  let prefersAgentSettings = false;
  let activeLlmProfile: string | null = null;
  let llmProfileOverride: string | undefined;
  if (
    !isCloud &&
    profile?.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME &&
    profile.agent_kind === "openhands"
  ) {
    // The seeded OpenHands `default` launches via agent_settings so the Skills
    // page selection (buildAgentContext) applies: a profile launch gets the
    // server's whole skill catalog minus a deny-list. Per-profile fields saved
    // on `default` therefore don't apply on local launches.
    //
    // Scoped to OpenHands: global agent_settings is stale when an ACP profile
    // is active. Scoped to local: cloud always resolves `default` server-side.
    prefersAgentSettings = true;
  } else if (profile?.agent_kind === "openhands" && profile.llm_profile_ref) {
    const llm = await readLlmProfiles();
    // An unreadable list leaves the ref unvalidated; handled below.
    const llmProfileExists =
      llm?.profiles.some((p) => p.name === profile.llm_profile_ref) ?? false;
    activeLlmProfile = llm?.active_profile ?? null;
    // The home LLM pill shows and activates the account-wide active LLM
    // profile, never the pinned ref (#16539), so an implicit launch runs that
    // selection. An explicit pick keeps its pinned ref unless the ref is gone.
    const preferredLlmProfile =
      llmProfile ||
      (!agentProfileId || !llmProfileExists ? activeLlmProfile : null);
    const selectedLlmProfile =
      !isCloud &&
      preferredLlmProfile &&
      preferredLlmProfile !== profile.llm_profile_ref
        ? preferredLlmProfile
        : undefined;
    if (!llmProfileExists || selectedLlmProfile) {
      if (!isCloud && (await backendSupportsLaunchLlmOverride())) {
        // A dangling ref with nothing selected reaches the server, which
        // rejects it with an error the user can act on.
        llmProfileOverride = selectedLlmProfile;
      } else if (!llmProfileExists || !llmProfile) {
        // No per-launch override on this server: run agent_settings at the
        // cost of the profile's other fields. A requested LLM is dropped
        // instead, keeping the profile.
        if (!llmProfileExists) {
          console.warn(
            `Agent profile "${profile.name}" references missing ` +
              `LLM profile "${profile.llm_profile_ref}"; ` +
              "launching from agent_settings instead.",
          );
        }
        prefersAgentSettings = true;
      }
    }
  }

  // A legacy agent_settings launch carries no profile identity, so Agent
  // Server cannot enforce that profile's secret allow-list. Keep restricted
  // profiles on the server-resolved path; if the detail cannot be read, fail
  // closed and let Agent Server report the underlying error.
  let profileAllowsAgentSettings = true;
  if (prefersAgentSettings && profile) {
    try {
      const detail = await queryClient.fetchQuery({
        queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
          backendId,
          orgId,
          profile.name,
        ),
        queryFn: () => AgentProfilesService.getProfile(profile.name),
        ...AGENT_PROFILES_RETRY_OPTIONS,
        meta: { disableToast: true },
      });
      const secretRefs = (detail.profile as { secret_refs?: unknown })
        .secret_refs;
      profileAllowsAgentSettings = !Array.isArray(secretRefs);
    } catch {
      profileAllowsAgentSettings = false;
    }
  }

  if (
    !requestedAgentProfileId ||
    (prefersAgentSettings && profileAllowsAgentSettings)
  ) {
    // An agent_settings launch runs the active LLM profile.
    if (!activeLlmProfile && !isCloud) {
      activeLlmProfile = (await readLlmProfiles())?.active_profile ?? null;
    }
    return { llmProfile: activeLlmProfile ?? undefined };
  }
  // Profile summaries omit ``acp_server``. Fetch the ACP profile detail so the
  // Claude skill overlay can tell Claude Code from Codex/custom without
  // launching the wrong overlay (#16905). Failures are ignored: conversation
  // start must not block on a detail lookup.
  let agentProfileAcpServer: string | null | undefined;
  let agentProfileAcpCommand: string | readonly string[] | null | undefined;
  if (!isCloud && profile?.agent_kind === "acp") {
    try {
      const detail = await queryClient.ensureQueryData({
        queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
          backendId,
          orgId,
          profile.name,
        ),
        queryFn: () => AgentProfilesService.getProfile(profile.name),
        ...AGENT_PROFILES_RETRY_OPTIONS,
      });
      const detailProfile = detail?.profile;
      if (detailProfile?.agent_kind === "acp") {
        agentProfileAcpServer = detailProfile.acp_server ?? null;
        agentProfileAcpCommand = detailProfile.acp_command ?? null;
      }
    } catch {
      console.warn(
        `Could not load ACP profile "${profile.name}" ` +
          "for Canvas skill projection; launching without the overlay.",
      );
    }
  }

  return {
    agentProfileId: requestedAgentProfileId,
    agentProfileKind: profile?.agent_kind,
    llmProfileOverride,
    llmProfile:
      profile?.agent_kind === "openhands"
        ? (llmProfileOverride ?? profile.llm_profile_ref ?? undefined)
        : undefined,
    agentProfileAcpServer,
    agentProfileAcpCommand,
  };
}

/** Create a conversation from the resolved launch and record the LLM it runs. */
export async function launchConversation(
  { agentProfileId, llmProfile, ...context }: ResolveConversationLaunchOptions,
  options: CreateConversationOptions,
): Promise<AppConversationStartTask> {
  const launch = await resolveConversationLaunch({
    ...context,
    agentProfileId,
    llmProfile,
  });
  const task = await AgentServerConversationService.createConversation({
    ...options,
    ...launchProfileOptions(launch),
  });
  // The chat LLM picker names this profile even when several share a model
  // (#1082). Cloud conversations don't use local LLM profiles.
  if (!context.isCloud && task.app_conversation_id && launch.llmProfile) {
    mergeStoredConversationMetadata(task.app_conversation_id, {
      active_profile: launch.llmProfile,
    });
  }
  return task;
}

function launchProfileOptions(
  launch: ConversationLaunch,
): Partial<CreateConversationOptions> {
  if (!launch.agentProfileId) return {};
  return {
    agentProfileId: launch.agentProfileId,
    agentProfileKind: launch.agentProfileKind,
    // Title generation uses the same LLM as the agent (#16885).
    agentLlmProfileRef: launch.llmProfile,
    ...(launch.llmProfileOverride
      ? { agentLlmProfileOverride: launch.llmProfileOverride }
      : {}),
    agentProfileAcpServer: launch.agentProfileAcpServer,
    agentProfileAcpCommand: launch.agentProfileAcpCommand,
  };
}
