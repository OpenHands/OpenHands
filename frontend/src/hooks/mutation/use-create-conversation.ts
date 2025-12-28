import { useMutation, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { SuggestedTask } from "#/utils/types";
import { Provider } from "#/types/settings";
import { CreateMicroagent, Conversation } from "#/api/open-hands.types";
import { useTracking } from "#/hooks/use-tracking";
import { useSettings } from "#/hooks/query/use-settings";

interface CreateConversationVariables {
  query?: string;
  repository?: {
    name: string;
    gitProvider: Provider;
    branch?: string;
  };
  suggestedTask?: SuggestedTask;
  conversationInstructions?: string;
  createMicroagent?: CreateMicroagent;
  parentConversationId?: string;
  agentType?: "default" | "plan";
}

// Response type that combines both V1 and legacy responses
interface CreateConversationResponse extends Partial<Conversation> {
  conversation_id: string;
  session_api_key: string | null;
  url: string | null;
  // V1 specific fields
  v1_task_id?: string;
  is_v1?: boolean;
}

export const useCreateConversation = () => {
  const queryClient = useQueryClient();
  const { trackConversationCreated } = useTracking();
  const { data: settings } = useSettings();

  return useMutation({
    mutationKey: ["create-conversation"],
    mutationFn: async (
      variables: CreateConversationVariables,
    ): Promise<CreateConversationResponse> => {
      const {
        query,
        repository,
        suggestedTask,
        conversationInstructions,
        createMicroagent,
        parentConversationId,
        agentType,
      } = variables;

      const useV1 = !!settings?.v1_enabled && !createMicroagent;

      if (useV1) {
        // Use V1 API - creates a conversation start task
        const startTask = await V1ConversationService.createConversation(
          repository?.name,
          repository?.gitProvider,
          query,
          repository?.branch,
          conversationInstructions,
          undefined, // trigger - will be set by backend
          parentConversationId,
          agentType,
        );

        // Return a special task ID that the frontend will recognize
        // Format: "task-{uuid}" so the conversation screen can poll the task
        // Once the task is ready, it will navigate to the actual conversation ID
        return {
          conversation_id: `task-${startTask.id}`,
          session_api_key: null,
          url: startTask.agent_server_url,
          v1_task_id: startTask.id,
          is_v1: true,
        };
      }

      // Use legacy API
      const conversation = await ConversationService.createConversation(
        repository?.name,
        repository?.gitProvider,
        query,
        suggestedTask,
        repository?.branch,
        conversationInstructions,
        createMicroagent,
      );

      return {
        ...conversation,
        is_v1: false,
      };
    },
    onSuccess: async (_, { repository }) => {
      trackConversationCreated({
        hasRepository: !!repository,
      });

      queryClient.removeQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
