import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService, {
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  AGENT_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { syncCloudAgentSettings } from "./sync-cloud-agent-settings";

export function useSaveAgentProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      name,
      profile,
    }: {
      name: string;
      profile: AgentProfileSaveInput;
    }) => AgentProfilesService.saveProfile(name, profile),
    onSuccess: async (_response, { name }) => {
      const settingsSynced = await syncCloudAgentSettings(name);
      // Prefix match invalidates every backend/org-suffixed list key.
      await queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
      // Only a Cloud save of the active profile rewrites agent_settings.
      if (settingsSynced) {
        await queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        });
      }
    },
    // Consumers handle errors with try-catch + targeted toasts (409/422).
    meta: { disableToast: true },
  });
}
