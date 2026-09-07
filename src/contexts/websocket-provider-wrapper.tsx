import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { findPlannerConversationId } from "#/utils/plan-file";

interface WebSocketProviderWrapperProps {
  children: React.ReactNode;
  conversationId: string;
}

export function WebSocketProviderWrapper({
  children,
  conversationId,
}: WebSocketProviderWrapperProps) {
  const { data: conversation } = useActiveConversation();
  const { backend } = useActiveBackend();
  const isLocalBackend = backend.kind !== "cloud";
  const localPlanningConversationId = useConversationStore(
    (state) => state.localPlanningConversationId,
  );

  // `localPlanningConversationId` is a single unscoped Zustand field. Right
  // after switching conversations it can still hold the *previous*
  // conversation's planner id, since the reset effect in
  // routes/conversation.tsx and `useActiveConversation()`'s refetch both lag
  // this render — only trust it once `conversation` matches `conversationId`.
  const isConversationDataFresh = conversation?.id === conversationId;
  const trustedLocalPlanningConversationId = isConversationDataFresh
    ? localPlanningConversationId
    : null;

  // Candidate ids to resolve. On local backends `sub_conversation_ids` is the
  // generic (untyped) child list, so fetch it below to find the
  // `plannerparent`-tagged entry; cloud has no such ambiguity. Memoized: a
  // fresh array literal each render would re-fire ConversationWebSocketProvider's
  // reference-keyed effects and wipe the pending PLAN.md read.
  const candidateConversationIds = React.useMemo(() => {
    if (!isLocalBackend) {
      return conversation?.sub_conversation_ids ?? [];
    }
    if (
      conversation?.sub_conversation_ids &&
      conversation.sub_conversation_ids.length > 0
    ) {
      return conversation.sub_conversation_ids;
    }
    return trustedLocalPlanningConversationId
      ? [trustedLocalPlanningConversationId]
      : [];
  }, [
    isLocalBackend,
    conversation?.sub_conversation_ids,
    trustedLocalPlanningConversationId,
  ]);
  const { data: subConversations } = useSubConversations(
    candidateConversationIds,
  );

  // Identify the planner via the `plannerparent` tag rather than list
  // position — an unrelated child must never be adopted as the planner.
  // Kept as its own memo (a primitive) rather than inlined below: `subConversations`
  // gets a new array reference on every refetch even when the planner id
  // hasn't changed, and planningConversationIds must not rebuild its own
  // array in that case — see candidateConversationIds above.
  const plannerConversationId = React.useMemo(() => {
    if (!isLocalBackend) return null;
    return findPlannerConversationId(subConversations, conversation?.id);
  }, [isLocalBackend, subConversations, conversation?.id]);

  const planningConversationIds = React.useMemo(() => {
    if (!isLocalBackend) return candidateConversationIds;
    if (plannerConversationId) return [plannerConversationId];
    // Tag data hasn't resolved yet — bridge with the verified store id (see
    // `trustedLocalPlanningConversationId` above), otherwise stay empty
    // rather than guessing an untagged child is the planner.
    return trustedLocalPlanningConversationId
      ? [trustedLocalPlanningConversationId]
      : [];
  }, [
    isLocalBackend,
    candidateConversationIds,
    plannerConversationId,
    trustedLocalPlanningConversationId,
  ]);

  const filteredSubConversations = subConversations?.filter(
    (subConversation): subConversation is AppConversation =>
      subConversation !== null &&
      planningConversationIds.includes(subConversation.id),
  );

  // Suppress the URL while the cloud sandbox is PAUSED — it still points at
  // the old (now-rejecting) host until the sandbox resumes.
  const conversationUrl =
    conversation?.sandbox_status === "PAUSED"
      ? null
      : conversation?.conversation_url;

  return (
    <ConversationWebSocketProvider
      conversationId={conversationId}
      conversationUrl={conversationUrl}
      sessionApiKey={conversation?.session_api_key}
      subConversationIds={planningConversationIds}
      subConversations={filteredSubConversations}
    >
      {children}
    </ConversationWebSocketProvider>
  );
}
