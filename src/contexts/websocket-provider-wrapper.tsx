import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { findPhasePlannerConversationId } from "#/utils/plan-file";
import { isArchivedSandboxStatus } from "#/utils/conversation-archive-status";

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
  const conversationMode = useConversationStore(
    (state) => state.conversationMode,
  );
  const activeDeepPlanPhase = useConversationStore(
    (state) => state.deepPlan.activePhase,
  );
  const deepPlanPlannerPhase = useConversationStore(
    (state) => state.deepPlanPlannerPhase,
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

  // Deep Planning creates one planner per phase, each pinned to that phase's
  // document; `plan` mode keeps the single untagged planner. Resolve the one
  // that matches the phase the user is in, so the socket and the PLAN.md read
  // follow the phase instead of always hitting the first planner.
  const deepPlanPhaseToResolve =
    conversationMode === "deep-plan" ? activeDeepPlanPhase : null;

  // Identify the planner via the `plannerparent` tag rather than list
  // position — an unrelated child must never be adopted as the planner.
  // Kept as its own memo (a primitive) rather than inlined below: `subConversations`
  // gets a new array reference on every refetch even when the planner id
  // hasn't changed, and planningConversationIds must not rebuild its own
  // array in that case — see candidateConversationIds above.
  const plannerConversationId = React.useMemo(() => {
    if (!isLocalBackend) return null;
    const phasePlanner = findPhasePlannerConversationId(
      subConversations,
      conversation?.id,
      deepPlanPhaseToResolve,
    );
    if (phasePlanner) return phasePlanner;
    // In deep-plan mode before a phase planner exists (or while the tag data
    // has not resolved), fall back to the store id so the planner is still
    // reachable; `plan` mode uses the plain untagged planner.
    if (deepPlanPhaseToResolve) {
      return deepPlanPlannerPhase === deepPlanPhaseToResolve
        ? trustedLocalPlanningConversationId
        : null;
    }
    // Plain plan mode: the untagged planner, never one of the per-phase ones.
    return findPhasePlannerConversationId(
      subConversations,
      conversation?.id,
      null,
    );
  }, [
    isLocalBackend,
    subConversations,
    conversation?.id,
    deepPlanPhaseToResolve,
    deepPlanPlannerPhase,
    trustedLocalPlanningConversationId,
  ]);

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

  // A paused or archived runtime cannot accept a WebSocket connection. Its
  // persisted event history remains available through the backend API.
  const conversationUrl =
    conversation?.sandbox_status === "PAUSED" ||
    isArchivedSandboxStatus(conversation?.sandbox_status)
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
