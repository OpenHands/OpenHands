import { useQuery } from "@tanstack/react-query";
import React from "react";
import { useConversationId } from "#/hooks/use-conversation-id";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useRuntimeIsReady } from "../use-runtime-is-ready";
import { useActiveConversation } from "./use-active-conversation";

/**
 * @deprecated This hook is for V0 conversations only. Use useUnifiedConversationConfig instead,
 * or useV1ConversationConfig once we fully migrate to V1.
 */
export const useV0ConversationConfig = () => {
  const { conversationId } = useConversationId();
  const runtimeIsReady = useRuntimeIsReady();

  const query = useQuery({
    queryKey: ["v0_conversation_config", conversationId],
    queryFn: () => {
      if (!conversationId) throw new Error("No conversation ID");
      return ConversationService.getRuntimeId(conversationId);
    },
    enabled: runtimeIsReady && !!conversationId,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  React.useEffect(() => {
    if (query.data) {
      const { runtime_id: runtimeId } = query.data;

      // eslint-disable-next-line no-console
      console.log(
        "Runtime ID: %c%s",
        "background: #444; color: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 4px;",
        runtimeId,
      );
    }
  }, [query.data]);

  return query;
};

export const useV1ConversationConfig = () => {
  const { conversationId } = useConversationId();
  const runtimeIsReady = useRuntimeIsReady();

  const query = useQuery({
    queryKey: ["v1_conversation_config", conversationId],
    queryFn: () => {
      if (!conversationId) throw new Error("No conversation ID");
      return V1ConversationService.getConversationConfig(conversationId);
    },
    enabled: runtimeIsReady && !!conversationId,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  React.useEffect(() => {
    if (query.data) {
      const { runtime_id: runtimeId } = query.data;

      // eslint-disable-next-line no-console
      console.log(
        "Runtime ID: %c%s",
        "background: #444; color: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 4px;",
        runtimeId,
      );
    }
  }, [query.data]);

  return query;
};

/**
 * Unified hook that switches between V0 and V1 conversation config endpoints based on conversation version.
 *
 * @temporary This hook is temporary during the V0 to V1 migration period.
 * Once we fully migrate to V1, all code should use useV1ConversationConfig directly.
 */
export const useUnifiedConversationConfig = () => {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const isV1Conversation = conversation?.conversation_version === "V1";

  const query = useQuery({
    queryKey: ["conversation_config", conversationId, isV1Conversation],
    queryFn: () => {
      if (!conversationId) throw new Error("No conversation ID");

      if (isV1Conversation) {
        return V1ConversationService.getConversationConfig(conversationId);
      }
      return ConversationService.getRuntimeId(conversationId);
    },
    enabled: runtimeIsReady && !!conversationId && conversation !== undefined,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  React.useEffect(() => {
    if (query.data) {
      const { runtime_id: runtimeId } = query.data;

      // eslint-disable-next-line no-console
      console.log(
        "Runtime ID: %c%s",
        "background: #444; color: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 4px;",
        runtimeId,
      );
    }
  }, [query.data]);

  return query;
};

// Keep the old export name for backward compatibility (uses unified approach)
export const useConversationConfig = useUnifiedConversationConfig;
