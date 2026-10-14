import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { StoredOpenHandsProfile } from "#/api/agent-profiles-service/profile-behavior";
import MetaProfilesService from "#/api/meta-profiles-service/meta-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_PROFILES_QUERY_KEYS,
  META_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

/** Detach a router from every agent profile using it; a launch refuses a missing one. */
async function clearRouterFromAgentProfiles(name: string): Promise<void> {
  const { profiles } = await AgentProfilesService.listProfiles();
  const details = await Promise.all(
    profiles
      .filter((profile) => profile.agent_kind === "openhands")
      .map((profile) => AgentProfilesService.getProfile(profile.name)),
  );
  await Promise.all(
    details
      .filter(
        ({ profile }) =>
          (profile as StoredOpenHandsProfile).meta_profile_ref === name,
      )
      .map(({ name: profileName, profile }) =>
        AgentProfilesService.saveProfile(
          profileName,
          mergeAgentProfileSaveInput(profile, {
            agent_kind: "openhands",
            meta_profile_ref: null,
            enable_classify_and_switch_llm_tool: false,
          } as Parameters<typeof mergeAgentProfileSaveInput>[1]),
        ),
      ),
  );
}

export function useDeleteMetaProfile() {
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();

  return useMutation({
    mutationFn: async (name: string) => {
      if (backend.kind === "local") await clearRouterFromAgentProfiles(name);
      return MetaProfilesService.deleteMetaProfile(name);
    },
    onSuccess: async () => {
      // Deleting the *active* meta-profile clears ``active_meta_profile`` in
      // settings, so refresh the settings caches too — mirroring activation.
      SettingsService.invalidateCache();
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: META_PROFILES_QUERY_KEYS.all,
        }),
        queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        }),
        queryClient.invalidateQueries({
          queryKey: AGENT_PROFILES_QUERY_KEYS.all,
        }),
      ]);
    },
    // Consumers handle errors with try-catch and manual toasts; disable global toast
    meta: { disableToast: true },
  });
}
