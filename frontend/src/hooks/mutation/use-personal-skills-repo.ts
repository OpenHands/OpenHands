import { useMutation, useQueryClient } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";

export const useSetPersonalSkillsRepo = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (repoUrl: string) =>
      SettingsService.setPersonalSkillsRepo(repoUrl),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["settings"] });
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
      await queryClient.invalidateQueries({ queryKey: ["settings"] });
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
      await queryClient.invalidateQueries({ queryKey: ["settings"] });
      await queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
    meta: { disableToast: true },
  });
};
