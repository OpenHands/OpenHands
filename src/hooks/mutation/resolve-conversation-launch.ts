import type { QueryClient } from "@tanstack/react-query";
import type { AgentKind } from "@openhands/typescript-client";
import AgentProfilesService, {
  type AgentProfileListResponse,
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { profileHasRouter } from "#/api/agent-profiles-service/profile-behavior";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { assertSubscriptionAuthReady } from "#/api/agent-server-adapter";
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
import { ensureDefaultAgentProfileMigrated } from "#/hooks/use-migrate-default-agent-profile";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";

export interface ConversationLaunch {
  /** Unset only on cloud, which launches its own default without one. */
  agentProfileId?: string;
  agentProfileKind?: AgentKind;
  /** Runs the agent profile with this LLM profile instead of its own. */
  llmProfileOverride?: string;
  /** The LLM profile the launch runs, when Canvas can tell. */
  llmProfile?: string;
  /** Local only: the launched profile carries a Model Router. */
  profileHasRouter?: boolean;
  /** Local only: the launched ACP profile's server, for the conversation tag. */
  acpServer?: string;
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

/** Decide which agent profile and LLM a new conversation launches with. */
export async function resolveConversationLaunch(
  options: ResolveConversationLaunchOptions,
): Promise<ConversationLaunch> {
  const { queryClient, backendId, orgId, isCloud, agentProfileId, llmProfile } =
    options;
  if (!isCloud) {
    // `default` must carry the global settings it used to launch from before
    // it launches by id. A failed migration retries on the next launch; this
    // one still runs the profile as stored.
    await ensureDefaultAgentProfileMigrated(queryClient, backendId).catch(
      (error) => console.warn("Could not migrate the default profile", error),
    );
  }
  // Await the list from the shared query cache: a send fired before the home
  // query resolves must still launch from the active profile.
  const agentProfiles: AgentProfileListResponse =
    await queryClient.ensureQueryData({
      queryKey: [...AGENT_PROFILES_QUERY_KEYS.all, backendId, orgId],
      queryFn: AgentProfilesService.listProfiles,
      ...AGENT_PROFILES_RETRY_OPTIONS,
      // A failure here rejects the launch, which reports it.
      meta: { disableToast: true },
    });

  const requestedAgentProfileId =
    agentProfileId ?? agentProfiles.active_agent_profile_id ?? undefined;
  if (!requestedAgentProfileId) {
    if (isCloud) return {};
    throw new Error(i18n.t(I18nKey.CONVERSATION$NO_ACTIVE_AGENT_PROFILE));
  }
  const profile = agentProfiles.profiles?.find(
    (p) => p.id === requestedAgentProfileId,
  );
  if (isCloud) {
    if (await cloudFallsBackToDefault(options, profile)) return {};
    return {
      agentProfileId: requestedAgentProfileId,
      agentProfileKind: profile?.agent_kind,
      llmProfile:
        profile?.agent_kind === "openhands"
          ? (profile.llm_profile_ref ?? undefined)
          : undefined,
    };
  }

  let llmProfileOverride: string | undefined;
  if (profile?.agent_kind === "openhands" && profile.llm_profile_ref) {
    // Await the list rather than a render snapshot: a send fired before the
    // LLM-profile query loads must still validate the ref.
    const llm = await queryClient
      .ensureQueryData({
        queryKey: [...LLM_PROFILES_QUERY_KEYS.all, backendId, orgId],
        queryFn: ProfilesService.listProfiles,
        retry: false,
        meta: { disableToast: true },
      })
      .catch(() => null);
    const llmProfileExists =
      llm?.profiles.some((p) => p.name === profile.llm_profile_ref) ?? false;
    // The home LLM pill shows and activates the account-wide active LLM
    // profile, never the pinned ref (#16539), so an implicit launch runs that
    // selection. An explicit pick keeps its pinned ref unless the ref is gone;
    // a dangling ref with nothing selected reaches the server, which rejects
    // it with an error the user can act on.
    const preferredLlmProfile =
      llmProfile ||
      (!agentProfileId || !llmProfileExists ? llm?.active_profile : null);
    if (
      preferredLlmProfile &&
      preferredLlmProfile !== profile.llm_profile_ref
    ) {
      llmProfileOverride = preferredLlmProfile;
    }
  }

  const detail = profile
    ? await queryClient
        .fetchQuery({
          queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
            backendId,
            orgId,
            profile.name,
          ),
          queryFn: () => AgentProfilesService.getProfile(profile.name),
          ...AGENT_PROFILES_RETRY_OPTIONS,
          meta: { disableToast: true },
        })
        // Router and tag are extras: the server still launches the profile.
        .catch(() => null)
    : null;
  const stored = detail?.profile;

  return {
    agentProfileId: requestedAgentProfileId,
    agentProfileKind: profile?.agent_kind,
    llmProfileOverride,
    llmProfile:
      profile?.agent_kind === "openhands"
        ? (llmProfileOverride ?? profile.llm_profile_ref ?? undefined)
        : undefined,
    profileHasRouter:
      stored?.agent_kind === "openhands" && profileHasRouter(stored),
    acpServer: stored?.agent_kind === "acp" ? stored.acp_server : undefined,
  };
}

/**
 * Cloud launches its own default instead of an OpenHands profile whose LLM
 * profile is gone, unless the profile restricts secrets, which only a profile
 * launch enforces.
 */
async function cloudFallsBackToDefault(
  { queryClient, backendId, orgId }: ResolveConversationLaunchOptions,
  profile: AgentProfileSummary | undefined,
): Promise<boolean> {
  if (profile?.agent_kind !== "openhands" || !profile.llm_profile_ref) {
    return false;
  }
  const llm = await queryClient
    .ensureQueryData({
      queryKey: [...LLM_PROFILES_QUERY_KEYS.all, backendId, orgId],
      queryFn: ProfilesService.listProfiles,
      retry: false,
      meta: { disableToast: true },
    })
    .catch(() => null);
  if (llm?.profiles.some((p) => p.name === profile.llm_profile_ref)) {
    return false;
  }
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
    return !Array.isArray(
      (detail.profile as { secret_refs?: unknown }).secret_refs,
    );
  } catch {
    return false;
  }
}

/** Fail before creating a conversation whose subscription LLM isn't connected. */
async function assertLaunchLlmReady(llmProfile: string): Promise<void> {
  let config: unknown;
  try {
    ({ config } = await ProfilesService.getProfile(llmProfile));
  } catch {
    return;
  }
  await assertSubscriptionAuthReady({ llm: config });
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
  if (!context.isCloud && launch.llmProfile) {
    await assertLaunchLlmReady(launch.llmProfile);
  }
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
    ...(launch.profileHasRouter ? { profileHasRouter: true } : {}),
    ...(launch.acpServer ? { acpServer: launch.acpServer } : {}),
  };
}
