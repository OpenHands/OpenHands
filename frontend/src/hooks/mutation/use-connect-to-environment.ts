import { useMutation, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

interface ConnectToEnvironmentVariables {
  environmentUrl: string;
  initialMessage?: string;
  agentType?: "default" | "plan";
}

interface ConnectToEnvironmentResponse {
  conversation_id: string;
  v1_task_id: string;
}

export const useConnectToEnvironment = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationKey: ["connect-to-environment"],
    mutationFn: async (
      variables: ConnectToEnvironmentVariables,
    ): Promise<ConnectToEnvironmentResponse> => {
      const { environmentUrl, initialMessage, agentType } = variables;

      const startTask = await V1ConversationService.connectToEnvironment(
        environmentUrl,
        initialMessage,
        agentType,
      );

      return {
        conversation_id: `task-${startTask.id}`,
        v1_task_id: startTask.id,
      };
    },
    onSuccess: async () => {
      queryClient.removeQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
