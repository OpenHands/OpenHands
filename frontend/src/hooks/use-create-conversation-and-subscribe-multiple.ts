import React from "react";
import { useQueries, type Query } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { AxiosError } from "axios";
import { useCreateConversation } from "./mutation/use-create-conversation";
import { useUserProviders } from "./use-user-providers";
import { useConversationSubscriptions } from "#/context/conversation-subscriptions-provider";
import { Provider } from "#/types/settings";
import { CreateMicroagent } from "#/api/open-hands.types";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import { renderConversationStartingToast } from "#/components/features/chat/microagent/microagent-status-toast";

interface ConversationData {
  conversationId: string;
  sessionApiKey: string | null;
  baseUrl: string;
  socketPath: string;
  onEventCallback?: (event: unknown, conversationId: string) => void;
}

/**
 * Custom hook to create a conversation and subscribe to it, supporting multiple subscriptions.
 * This version waits for conversation status to be "RUNNING" before establishing WebSocket connection.
 * Shows immediate toast feedback and polls conversation status until ready.
 */
export const useCreateConversationAndSubscribeMultiple = () => {
  const { mutate: createConversation, isPending } = useCreateConversation();
  const { providers } = useUserProviders();
  const {
    subscribeToConversation,
    unsubscribeFromConversation,
    isSubscribedToConversation,
    activeConversationIds,
  } = useConversationSubscriptions();

  // Store conversation data immediately after creation
  const [createdConversations, setCreatedConversations] = React.useState<
    Record<string, ConversationData>
  >({});

  // Get conversation IDs that need polling
  const conversationIdsToWatch = Object.keys(createdConversations);

  // Poll each conversation until it's ready
  const conversationQueries = useQueries({
    queries: conversationIdsToWatch.map((conversationId) => ({
      queryKey: ["conversation-ready-poll", conversationId],
      queryFn: async () => {
        const result = await V1ConversationService.batchGetAppConversations([
          conversationId,
        ]);
        return result[0] || null;
      },
      enabled: !!conversationId,
      refetchInterval: (query: Query<V1AppConversation | null, AxiosError>) => {
        const sandboxStatus = query.state.data?.sandbox_status;
        if (sandboxStatus === "STARTING") {
          return 3000; // Poll every 3 seconds while STARTING
        }
        return false; // Stop polling once not STARTING
      },
      retry: false,
    })),
  });

  // Extract stable values from dependency array
  const queryStatuses = conversationQueries.map(
    (query) => query.data?.sandbox_status,
  );
  const queryDataExists = conversationQueries.map((query) => !!query.data);

  // Effect to handle subscription when conversations are ready
  React.useEffect(() => {
    conversationQueries.forEach((query, index) => {
      const conversationId = conversationIdsToWatch[index];
      const conversationData = createdConversations[conversationId];
      const conversation = query.data;

      // Check if conversation is ready (sandbox is running)
      if (conversation?.sandbox_status === "RUNNING" && conversationData) {
        const {
          sandbox_status: sandboxStatus,
          conversation_url: url,
          session_api_key: sessionApiKey,
        } = conversation;

        let { baseUrl } = conversationData;
        if (url && !url.startsWith("/")) {
          baseUrl = new URL(url).host;
        }

        if (sandboxStatus === "RUNNING") {
          // Conversation is ready - subscribe to WebSocket
          subscribeToConversation({
            conversationId,
            sessionApiKey,
            providersSet: providers as Provider[],
            baseUrl,
            socketPath: conversationData.socketPath,
            onEvent: conversationData.onEventCallback,
          });

          // Remove from created conversations (cleanup)
          setCreatedConversations((prev) => {
            const newCreated = { ...prev };
            delete newCreated[conversationId];
            return newCreated;
          });
        } else if (sandboxStatus === "MISSING") {
          // Dismiss the starting toast
          toast.dismiss(`starting-${conversationId}`);

          // Remove from created conversations (cleanup)
          setCreatedConversations((prev) => {
            const newCreated = { ...prev };
            delete newCreated[conversationId];
            return newCreated;
          });
        }
      }
    });
  }, [
    queryStatuses,
    queryDataExists,
    conversationIdsToWatch,
    createdConversations,
    subscribeToConversation,
    providers,
  ]);

  const createConversationAndSubscribe = React.useCallback(
    ({
      query,
      conversationInstructions,
      repository,
      createMicroagent,
      onSuccessCallback,
      onEventCallback,
    }: {
      query: string;
      conversationInstructions: string;
      repository: {
        name: string;
        branch?: string;
        gitProvider: Provider;
      };
      createMicroagent?: CreateMicroagent;
      onSuccessCallback?: (conversationId: string) => void;
      onEventCallback?: (event: unknown, conversationId: string) => void;
    }) => {
      createConversation(
        {
          query,
          conversationInstructions,
          repository,
          createMicroagent,
        },
        {
          onSuccess: (data) => {
            // Show immediate toast to let user know something is happening
            renderConversationStartingToast(data.conversation_id);

            // Call the success callback immediately
            if (onSuccessCallback) {
              onSuccessCallback(data.conversation_id);
            }

            // Only handle immediate post-creation tasks here
            let baseUrl = "";
            let socketPath: string;
            if (data?.url && !data.url.startsWith("/")) {
              const u = new URL(data.url);
              baseUrl = u.host;
              const pathBeforeApi =
                u.pathname.split("/api/conversations")[0] || "/";
              socketPath = `${pathBeforeApi.replace(/\/$/, "")}/socket.io`;
            } else {
              baseUrl =
                (import.meta.env.VITE_BACKEND_BASE_URL as string | undefined) ||
                window?.location.host;
              socketPath = "/socket.io";
            }

            // Store conversation data for polling and eventual subscription
            setCreatedConversations((prev) => ({
              ...prev,
              [data.conversation_id]: {
                conversationId: data.conversation_id,
                sessionApiKey: data.session_api_key,
                baseUrl,
                socketPath,
                onEventCallback,
              },
            }));
          },
        },
      );
    },
    [createConversation],
  );

  return {
    createConversationAndSubscribe,
    unsubscribeFromConversation,
    isSubscribedToConversation,
    activeConversationIds,
    isPending,
  };
};
