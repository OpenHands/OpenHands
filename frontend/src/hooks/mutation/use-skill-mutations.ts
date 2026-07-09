import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { I18nKey } from "#/i18n/declaration";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

export interface UseSkillMutationsReturn {
  saveDisabledSkills: ReturnType<
    typeof useMutation<unknown, AxiosError, string[]>
  >;
}

/**
 * Hook for skill-related mutations.
 * Handles saving disabled_skills state.
 */
export function useSkillMutations(
  options: { onSettled?: () => void } = {},
): UseSkillMutationsReturn {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  const saveDisabledSkills = useMutation({
    mutationFn: async (disabledSkills: string[]) => {
      await SettingsService.saveSettings({ disabled_skills: disabledSkills });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
    },
    onError: (error: AxiosError) => {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
      );
    },
    onSettled: options.onSettled,
    meta: { disableToast: true },
  });

  return { saveDisabledSkills };
}
