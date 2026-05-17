import { useMutation, useQueryClient } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

export const useSetPersonalSkillsRepo = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (repoUrl: string) =>
      SettingsService.setPersonalSkillsRepo(repoUrl),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.all,
      });
      await queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
    meta: { disableToast: true },
  });
};

export const useUpdatePersonalSkillsRepo = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => SettingsService.updatePersonalSkillsRepo(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.all,
      });
      await queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
    meta: { disableToast: true },
  });
};

export const useRemovePersonalSkillsRepo = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => SettingsService.removePersonalSkillsRepo(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.all,
      });
      await queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
    meta: { disableToast: true },
  });
};
