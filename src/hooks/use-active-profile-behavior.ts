import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { StoredOpenHandsProfile } from "#/api/agent-profiles-service/profile-behavior";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import { useActiveAgentProfile } from "#/hooks/use-active-agent-profile";
import {
  ensureDefaultAgentProfileMigrated,
  useMigrateDefaultAgentProfile,
} from "#/hooks/use-migrate-default-agent-profile";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
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
  /** The profile when it is an OpenHands agent and safe to edit. */
  profile: StoredOpenHandsProfile | null;
  /** The profile runs an ACP agent, which these settings don't reach. */
  isAcp: boolean;
  isLoading: boolean;
  /** The profile could not be loaded, or no profile is active. */
  isError: boolean;
  retry: () => void;
  save: (patch: ProfileBehaviorPatch) => Promise<void>;
  isSaving: boolean;
}

/**
 * A local agent profile — the active one unless `agentProfileId` names
 * another — and a save that merges a patch over its latest stored copy. The
 * profile is withheld until the one-time `default` migration has run, so no
 * edit is built from settings the migration would then overwrite.
 */
export function useActiveProfileBehavior(
  agentProfileId?: string | null,
): ActiveProfileBehavior {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const { activeProfile, isLoading: activeLoading } = useActiveAgentProfile();
  const {
    data: profiles,
    isLoading: listLoading,
    refetch: refetchProfiles,
  } = useAgentProfiles();
  const summary = agentProfileId
    ? (profiles?.profiles?.find((p) => p.id === agentProfileId) ?? null)
    : activeProfile;
  const profileName = isLocal ? (summary?.name ?? null) : null;
  const migration = useMigrateDefaultAgentProfile();
  const migrated = migration.isSuccess;

  const detail = useQuery({
    queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
      backend.id,
      orgId,
      profileName ?? "",
    ),
    queryFn: () => AgentProfilesService.getProfile(profileName!),
    enabled: profileName != null && migrated,
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

  const stored = migrated ? detail.data?.profile : undefined;
  const listSettled = !activeLoading && !listLoading;
  return {
    isLocal,
    profileName,
    profile:
      stored?.agent_kind === "openhands"
        ? (stored as StoredOpenHandsProfile)
        : null,
    isAcp: summary?.agent_kind === "acp",
    isLoading:
      isLocal &&
      (!listSettled || migration.isPending || (migrated && detail.isLoading)),
    isError:
      isLocal &&
      (migration.isError ||
        detail.isError ||
        (listSettled && profileName === null)),
    retry: () => {
      if (migration.isError) migration.refetch();
      if (detail.isError) detail.refetch();
      if (profileName === null) refetchProfiles();
    },
    save: async (patch) => {
      await mutation.mutateAsync(patch);
    },
    isSaving: mutation.isPending,
  };
}
