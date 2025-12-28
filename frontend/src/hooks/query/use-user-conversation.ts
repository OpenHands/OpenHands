/* eslint-disable @typescript-eslint/no-explicit-any */
import { Query, useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { Conversation } from "#/api/open-hands.types";

const FIVE_MINUTES = 1000 * 60 * 5;
const FIFTEEN_MINUTES = 1000 * 60 * 15;

type RefetchInterval = (
  query: Query<
    Conversation | null,
    AxiosError<unknown, any>,
    Conversation | null,
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

      // Use the legacy GET endpoint - it handles both V0 and V1 conversations
      // V1 conversations are automatically detected by UUID format and converted
      const conversation = await ConversationService.getConversation(cid);
      return conversation;
    },
    enabled: !!cid,
    retry: false,
    refetchInterval,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
  });
