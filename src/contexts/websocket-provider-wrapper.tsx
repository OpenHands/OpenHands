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

  // Candidate ids to resolve. On local backends `sub_conversation_ids` is the
  // generic (untyped) child list, so fetch it below to find the
  // `plannerparent`-tagged entry. Cloud has no such ambiguity — the app-server
  // only ever attaches the planner here, so it's trusted directly.
  //
  // Stable reference across renders: ConversationWebSocketProvider keys effects
  // on `subConversationIds` (planning-history tracking + the deferred PLAN.md
  // read), so a fresh array literal each render re-fires them and wipes the
  // pending plan read — the local planner's PLAN.md would never surface.
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
    return localPlanningConversationId ? [localPlanningConversationId] : [];
  }, [
    isLocalBackend,
    conversation?.sub_conversation_ids,
    localPlanningConversationId,
  ]);
  const { data: subConversations } = useSubConversations(
    candidateConversationIds,
  );

  // `sub_conversation_ids` makes no promise that any entry (let alone index
  // 0) is the planner — identify it explicitly via the `plannerparent` tag
  // that local planner creation stamps, rather than assuming list position
  // implies type. A pre-existing, unrelated child conversation must never be
  // adopted as the planner and fed events/routing meant for it.
  const planningConversationIds = React.useMemo(() => {
    if (!isLocalBackend) return candidateConversationIds;
    const plannerConversationId = findPlannerConversationId(
      subConversations,
      conversation?.id,
    );
    if (plannerConversationId) return [plannerConversationId];
    // Tag data hasn't resolved yet (still loading, or no planner exists).
    // `localPlanningConversationId` is only ever set to a verified planner
    // id (freshly created, or restored via the same tag check in
    // useHandlePlanClick), so it's safe to bridge with synchronously;
    // otherwise stay empty rather than guessing that an untagged child is
    // the planner.
    return localPlanningConversationId ? [localPlanningConversationId] : [];
  }, [
    isLocalBackend,
    candidateConversationIds,
    subConversations,
    conversation?.id,
    localPlanningConversationId,
  ]);

  const filteredSubConversations = subConversations?.filter(
    (subConversation): subConversation is AppConversation =>
      subConversation !== null &&
      planningConversationIds.includes(subConversation.id),
  );

  // Don't pass a conversation URL to the WebSocket provider while the cloud
  // sandbox is PAUSED. The URL still points to the old sandbox host, which
  // rejects connections until the sandbox has fully resumed. Treating the URL
  // as absent here keeps wsUrl === null in ConversationWebSocketProvider, so
  // no connection is attempted until useActiveConversation detects the
  // transition out of PAUSED (via fast 3-second polling).
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
