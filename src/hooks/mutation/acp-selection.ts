import type { QueryClient } from "@tanstack/react-query";
import AgentProfilesService, {
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import {
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { ACP_SESSION_RUNTIME_QUERY_KEY } from "#/hooks/query/use-acp-session-models";
import { invalidateConversationQueries } from "./conversation-mutation-utils";

/** The ACP launch fields the home page picker can save. */
export type AcpSelection =
  { acp_model: string | null } | { acp_effort: string | null };

/**
 * Save a home-page ACP pick as the launch default.
 *
 * The target is the **active ACP AgentProfile**: profile launches resolve the
 * agent server-side purely from the stored profile (``agent_profile_id`` and
 * ``agent_settings`` are mutually exclusive launch sources), so writing
 * ``agent_settings`` would not affect them. Only when the profile list resolves
 * with no active ACP profile does the write fall back to the
 * ``agent_settings_diff`` a Cloud launch without a profile reads. A failed
 * profile discovery propagates instead of downgrading (#16523): an
 * active-profile launch ignores agent_settings, so persisting the pick there
 * would silently drop it.
 */
export async function persistHomeAcpSelection(
  queryClient: QueryClient,
  backendId: string,
  orgId: string | null | undefined,
  selection: AcpSelection,
): Promise<void> {
  // Same keys/retry policy as useCreateConversation's launch path, so a pick
  // can't disagree with what a subsequent send would launch.
  const agentProfiles: AgentProfileListResponse =
    await queryClient.ensureQueryData({
      queryKey: [...AGENT_PROFILES_QUERY_KEYS.all, backendId, orgId],
      queryFn: AgentProfilesService.listProfiles,
      ...AGENT_PROFILES_RETRY_OPTIONS,
    });
  const activeProfile = agentProfiles.profiles?.find(
    (profile) =>
      profile.id != null &&
      profile.id === agentProfiles.active_agent_profile_id,
  );

  if (activeProfile?.agent_kind === "acp") {
    // Merge over the stored profile so a one-field pick can't wipe the fields
    // this surface doesn't model (command, session mode, …).
    const detail = await queryClient.ensureQueryData({
      queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
        backendId,
        orgId,
        activeProfile.name,
      ),
      queryFn: () => AgentProfilesService.getProfile(activeProfile.name),
      ...AGENT_PROFILES_RETRY_OPTIONS,
    });
    await AgentProfilesService.saveProfile(
      activeProfile.name,
      mergeAgentProfileSaveInput(detail.profile, {
        agent_kind: "acp",
        ...selection,
      }),
    );
    return;
  }

  // The backend deep-merges ``agent_settings_diff`` into the existing
  // ``agent_settings`` dict, so a scalar diff updates only that field and
  // preserves the selected provider + command.
  await SettingsService.saveSettings({ agent_settings_diff: selection });
}

/** Refresh what an ACP pick changed: the conversation, or the launch default. */
export function refreshAfterAcpSelection(
  queryClient: QueryClient,
  conversationId: string | null,
): void {
  if (conversationId) {
    invalidateConversationQueries(queryClient, conversationId);
    queryClient.invalidateQueries({
      queryKey: [ACP_SESSION_RUNTIME_QUERY_KEY, conversationId],
    });
    return;
  }
  // Clear the stale settings cache so the next conversation-start reads the
  // newly saved default, and refetch so the home-page chip updates. The
  // agent-profiles prefix covers both the list and the detail entry rewritten.
  SettingsService.invalidateCache();
  queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.personal() });
  queryClient.invalidateQueries({ queryKey: AGENT_PROFILES_QUERY_KEYS.all });
}
