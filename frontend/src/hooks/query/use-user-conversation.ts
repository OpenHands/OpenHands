/* eslint-disable @typescript-eslint/no-explicit-any */
import { Query, useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import {
  getRateLimitRetryDelayMs,
  isRateLimitError,
} from "#/utils/rate-limit-retry";

const FIVE_MINUTES = 1000 * 60 * 5;
const FIFTEEN_MINUTES = 1000 * 60 * 15;

type RefetchInterval = (
  query: Query<
    V1AppConversation | null,
    AxiosError<unknown, any>,
    V1AppConversation | null,
    (string | null)[]
  >,
) => number;

export const useUserConversation = (
  cid: string | null,
  refetchInterval?: RefetchInterval,
) =>
  useQuery({
    queryKey: ["user", "conversation", cid],
    queryFn: async () => {
      if (!cid) return null;

      // Use the V1 batch API endpoint to get a single conversation
      const results = await V1ConversationService.batchGetAppConversations([
        cid,
      ]);
      return results[0] ?? null;
    },
    enabled: !!cid && !cid.startsWith("task-"),
    retry: (failureCount, error) => isRateLimitError(error) && failureCount < 3,
    retryDelay: (failureCount, error) =>
      getRateLimitRetryDelayMs(failureCount, error),
    refetchInterval,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
  });
