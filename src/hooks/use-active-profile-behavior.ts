import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { StoredOpenHandsProfile } from "#/api/agent-profiles-service/profile-behavior";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import { useActiveAgentProfile } from "#/hooks/use-active-agent-profile";
import { ensureDefaultAgentProfileMigrated } from "#/hooks/use-migrate-default-agent-profile";
import {
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
} from "#/hooks/query/query-keys";

export type ProfileBehaviorPatch = Partial<
  Omit<StoredOpenHandsProfile, "agent_kind" | "id" | "name" | "revision">
>;

export interface ActiveProfileBehavior {
  /** False on Cloud, where these settings stay global. */
  isLocal: boolean;
  profileName: string | null;
  /** The active profile when it is an OpenHands agent. */
  profile: StoredOpenHandsProfile | null;
  /** The active profile runs an ACP agent, which these settings don't reach. */
  isAcp: boolean;
  isLoading: boolean;
  save: (patch: ProfileBehaviorPatch) => Promise<void>;
  isSaving: boolean;
}

/**
 * The local agent profile that new conversations launch with, and a save that
 * merges a patch over its latest stored copy.
 */
export function useActiveProfileBehavior(): ActiveProfileBehavior {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const { activeProfile, isLoading: listLoading } = useActiveAgentProfile();
  const profileName = isLocal ? (activeProfile?.name ?? null) : null;

  const detail = useQuery({
    queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
      backend.id,
      orgId,
      profileName ?? "",
    ),
    queryFn: () => AgentProfilesService.getProfile(profileName!),
    enabled: profileName != null,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });

  const mutation = useMutation({
    mutationFn: async (patch: ProfileBehaviorPatch) => {
      if (!profileName) throw new Error("No agent profile is active.");
      // Saving before the one-time migration would let it overwrite the edit.
      await ensureDefaultAgentProfileMigrated(queryClient, backend.id);
      const { profile } = await AgentProfilesService.getProfile(profileName);
      if (profile.agent_kind !== "openhands") {
        throw new Error(
          `Agent profile "${profileName}" is not an OpenHands agent.`,
        );
      }
      await AgentProfilesService.saveProfile(
        profileName,
        mergeAgentProfileSaveInput(profile, {
          agent_kind: "openhands",
          ...patch,
        } as Parameters<typeof mergeAgentProfileSaveInput>[1]),
      );
    },
    onSettled: () =>
      queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      }),
    meta: { disableToast: true },
  });

  const stored = detail.data?.profile;
  return {
    isLocal,
    profileName,
    profile:
      stored?.agent_kind === "openhands"
        ? (stored as StoredOpenHandsProfile)
        : null,
    isAcp: activeProfile?.agent_kind === "acp",
    isLoading: isLocal && (listLoading || detail.isLoading),
    save: async (patch) => {
      await mutation.mutateAsync(patch);
    },
    isSaving: mutation.isPending,
  };
}
