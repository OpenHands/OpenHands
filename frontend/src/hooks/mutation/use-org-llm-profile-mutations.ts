import { useMutation, useQueryClient } from "@tanstack/react-query";
import OrgProfilesService, {
  SaveOrgLlmProfileRequest,
} from "#/api/organization-service/org-profiles-service.api";
import { ORG_LLM_PROFILES_QUERY_KEY } from "#/hooks/query/use-org-llm-profiles";
import { LLM_PROFILES_QUERY_KEY } from "#/hooks/query/use-llm-profiles";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

// The chat-layer switch button and /model command read the personal
// profiles query (/api/v1/settings/profiles), which surfaces org profiles
// in SaaS mode via SaasSettingsStore. Invalidate that cache too so those
// surfaces refresh after org-side CRUD without a full reload.
const invalidateProfileCaches = (
  queryClient: ReturnType<typeof useQueryClient>,
  orgId: string | null | undefined,
) => {
  queryClient.invalidateQueries({
    queryKey: [ORG_LLM_PROFILES_QUERY_KEY, orgId],
  });
  queryClient.invalidateQueries({
    queryKey: [LLM_PROFILES_QUERY_KEY],
  });
};

export function useSaveOrgLlmProfile(orgId: string | null | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      name,
      request,
    }: {
      name: string;
      request?: SaveOrgLlmProfileRequest;
    }) => {
      if (!orgId) throw new Error("Organization ID is required");
      await OrgProfilesService.saveProfile(orgId, name, request ?? {});
    },
    onSuccess: () => {
      invalidateProfileCaches(queryClient, orgId);
    },
  });
}

export function useDeleteOrgLlmProfile(orgId: string | null | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (name: string) => {
      if (!orgId) throw new Error("Organization ID is required");
      await OrgProfilesService.deleteProfile(orgId, name);
    },
    onSuccess: () => {
      invalidateProfileCaches(queryClient, orgId);
    },
  });
}

export function useActivateOrgLlmProfile(orgId: string | null | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (name: string) => {
      if (!orgId) throw new Error("Organization ID is required");
      await OrgProfilesService.activateProfile(orgId, name);
    },
    onSuccess: () => {
      invalidateProfileCaches(queryClient, orgId);
      // Activation writes the member's agent_settings_diff, whose effective
      // settings surface under the *personal* scope (not just "org"). Invalidate
      // all settings scopes — matching the personal useActivateLlmProfile hook —
      // so every settings surface refetches the new active LLM config.
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.all,
      });
    },
  });
}

export function useRenameOrgLlmProfile(orgId: string | null | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      name,
      newName,
    }: {
      name: string;
      newName: string;
    }) => {
      if (!orgId) throw new Error("Organization ID is required");
      await OrgProfilesService.renameProfile(orgId, name, newName);
    },
    onSuccess: () => {
      invalidateProfileCaches(queryClient, orgId);
    },
  });
}
