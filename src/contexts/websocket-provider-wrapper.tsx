import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  findPhasePlannerConversationId,
  isFallbackPlannerId,
} from "#/utils/plan-file";
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
    // has not resolved), fall back to the store id only when it was recorded
    // for *this* phase and its owner can be proven. A nil result here means the
    // active phase has no planner yet — sends must not fall back to another
    // phase's planner (which edits the wrong document), so the caller leaves
    // the target empty until the new planner appears.
    if (deepPlanPhaseToResolve) {
      return isFallbackPlannerId(
        subConversations,
        conversation?.id,
        trustedLocalPlanningConversationId,
        deepPlanPhaseToResolve,
        deepPlanPlannerPhase,
      )
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

  // Bridge candidate (a primitive, so `planningConversationIds` below keeps a
  // stable array reference across refetches that resolve to the same planner).
  // The store id is bridged only when it belongs to this resolve context (same
  // phase, proven owner); blindly bridging would send messages while the new
  // phase's planner is still provisioning into the *previous* phase's planner,
  // whose tool edits the wrong document. Otherwise there is no bridge: an
  // unresolved phase must not fall back to an unrelated planner.
  const bridgePlannerId =
    trustedLocalPlanningConversationId &&
    isFallbackPlannerId(
      subConversations,
      conversation?.id,
      trustedLocalPlanningConversationId,
      deepPlanPhaseToResolve,
      deepPlanPlannerPhase,
    )
      ? trustedLocalPlanningConversationId
      : null;

  const planningConversationIds = React.useMemo(() => {
    if (!isLocalBackend) return candidateConversationIds;
    if (plannerConversationId) return [plannerConversationId];
    return bridgePlannerId ? [bridgePlannerId] : [];
  }, [
    isLocalBackend,
    candidateConversationIds,
    plannerConversationId,
    bridgePlannerId,
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
