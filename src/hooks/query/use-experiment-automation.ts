import { useMutation, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { CreateExperimentAutomationRequest } from "#/types/experiment";
import { AUTOMATIONS_QUERY_KEY } from "./use-automations";

export function useCreateExperimentAutomation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateExperimentAutomationRequest) =>
      AutomationService.createExperimentAutomation(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUTOMATIONS_QUERY_KEY });
    },
  });
}
