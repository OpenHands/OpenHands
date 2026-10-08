import { useCallback, useEffect, type MouseEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import type { ConversationMode } from "#/stores/conversation-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";
import { useActiveBackend } from "#/contexts/active-backend-context";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import {
  CONVERSATION_QUERY_KEYS,
  LOCAL_PLANNER_MUTATION_KEYS,
} from "#/hooks/query/query-keys";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import {
  findPhasePlannerConversationId,
  findPlannerConversationId,
} from "#/utils/plan-file";
import { type DeepPlanPhaseId } from "#/utils/deep-plan";
import { deepPlanGuidance } from "#/utils/deep-plan-messages";
import { isPlanningMode } from "#/utils/conversation-mode";
import { invalidateConversationQueries } from "#/hooks/mutation/conversation-mutation-utils";

/**
 * Phases whose planner creation is currently in flight, keyed by parent then
 * phase. `useHandlePlanClick` mounts in several places (chat input, agent
 * button, planner tab), so their provisioning effects can fire in the same tick
 * — before any of their React Query mutations reports `isPending`. React Query
 * only de-dupes mutations sharing a `mutationKey`; ours is static, so this
 * module-level set is what stops two concurrent effects from creating two
 * planners for the same phase.
 */
const inFlightPhasePlannerCreations = new Set<string>();

function phasePlannerCreationKey(
  parentConversationId: string,
  phase: DeepPlanPhaseId | null,
): string {
  return `${parentConversationId}::${phase ?? "plan"}`;
}

function useCreateLocalPlanningConversationMutation(options: {
  onCreated: (
    planningConversationId: string,
    deepPlanPhase: DeepPlanPhaseId | null,
  ) => void;
  onInitialized: () => void;
  onFailed: () => void;
}) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationKey: LOCAL_PLANNER_MUTATION_KEYS.create,
    mutationFn: (variables: {
      parentConversationId: string;
      initialMessage?: string;
      deepPlanPhase?: DeepPlanPhaseId | null;
      deepPlanGuidance?: string | null;
    }) =>
      AgentServerConversationService.createLocalPlanningConversation(
        variables.parentConversationId,
        variables.initialMessage,
        variables.deepPlanPhase,
        variables.deepPlanGuidance,
      ),
    onSuccess: (planningConversation, variables) => {
      inFlightPhasePlannerCreations.delete(
        phasePlannerCreationKey(
          variables.parentConversationId,
          variables.deepPlanPhase ?? null,
        ),
      );
      options.onCreated(
        planningConversation.id,
        variables.deepPlanPhase ?? null,
      );
      invalidateConversationQueries(
        queryClient,
        variables.parentConversationId,
      );
      queryClient.invalidateQueries({
        queryKey: CONVERSATION_QUERY_KEYS.subConversations,
      });
      options.onInitialized();
    },
    onError: (error, variables) => {
      inFlightPhasePlannerCreations.delete(
        phasePlannerCreationKey(
          variables.parentConversationId,
          variables.deepPlanPhase ?? null,
        ),
      );
      options.onFailed();
    },
  });
}

function restorePlanningConversationIds(options: {
  conversationId: string;
  serverPlanningConversationId: string | null;
  subConversationTaskId: string | null;
  localPlanningConversationId: string | null;
  setSubConversationTaskId: (taskId: string | null) => void;
  setLocalPlanningConversationId: (conversationId: string | null) => void;
}) {
  const storedState = getConversationState(options.conversationId);
  if (storedState.subConversationTaskId && !options.subConversationTaskId) {
    options.setSubConversationTaskId(storedState.subConversationTaskId);
  }

  // Server first: `sub_conversation_ids` is derived by the agent-server from
  // the planner's `parent_conversation_id`, so it survives cleared site data
  // and follows the user to another browser. The localStorage hint is only the
  // fallback for agent-servers older than 1.37.1, which drop the parent link.
  const restoredId =
    options.serverPlanningConversationId ??
    getStoredConversationMetadata(options.conversationId)
      ?.local_planning_conversation_id ??
    null;

  if (restoredId && restoredId !== options.localPlanningConversationId) {
    options.setLocalPlanningConversationId(restoredId);
  }
}

/**
 * Custom hook that encapsulates the logic for handling plan creation.
 * Returns a function that can be called to create a plan conversation and
 * the pending state of the conversation creation.
 *
 * @returns An object containing handlePlanClick function and isCreatingConversation boolean
 */
export const useHandlePlanClick = () => {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const {
    setConversationMode,
    conversationMode,
    setSubConversationTaskId,
    subConversationTaskId,
    setLocalPlanningConversationId,
    localPlanningConversationId,
    deepPlanPlannerPhase,
  } = useConversationStore();
  const { data: conversation } = useActiveConversation();
  const { mutate: createConversation, isPending: isCreatingCloudConversation } =
    useCreateConversation();
  const {
    mutate: createLocalPlanningConversation,
    isPending: isCreatingLocalPlanningConversation,
  } = useCreateLocalPlanningConversationMutation({
    onCreated: setLocalPlanningConversationId,
    onInitialized: () => {
      displaySuccessToast(
        t(I18nKey.PLANNING_AGENTT$PLANNING_AGENT_INITIALIZED),
      );
    },
    // handlePlanClick sets conversationMode("plan") before this mutation
    // starts. On failure, back out of that mode instead of stranding the
    // user in plan mode with no planner to talk to.
    onFailed: () => {
      setConversationMode("code");
      displayErrorToast(t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION));
    },
  });

  // On local backends the agent-server reports the planner helper back on the
  // parent's `sub_conversation_ids` (it was created with
  // `parent_conversation_id`), so that is the authoritative handle. Cloud
  // sub-conversations are driven by their own task/socket plumbing.
  const isLocalBackend = backend.kind !== "cloud";
  const activeDeepPlanPhase = useConversationStore(
    (state) => state.deepPlan.activePhase,
  );
  const { data: rawSubConversations } = useSubConversations(
    isLocalBackend ? conversation?.sub_conversation_ids : undefined,
  );

  const serverPlanningConversationId = isLocalBackend
    ? findPlannerConversationId(rawSubConversations, conversation?.id)
    : null;

  // Whether the current Deep Planning phase already has its own planner. Each
  // phase's planner is pinned to that phase's document, so "has a planner" is
  // per-phase, not per-conversation. The Implementation phase runs in the code
  // agent (`isPlanningMode` excludes it), so it needs no planner.
  const hasPhasePlanner =
    isLocalBackend &&
    activeDeepPlanPhase !== null &&
    isPlanningMode("deep-plan", activeDeepPlanPhase)
      ? !!findPhasePlannerConversationId(
          rawSubConversations,
          conversation?.id,
          activeDeepPlanPhase,
        ) || deepPlanPlannerPhase === activeDeepPlanPhase
      : false;

  // Restore planning conversation ids on conversation load. This handles page
  // refreshes while cloud or local planning conversation creation is in
  // progress, and recovers the local planner after browser storage is lost.
  useEffect(() => {
    if (!conversation?.id) return;
    // Deep Planning keeps one planner per phase, resolved from each planner's
    // phase tag, so the single-planner restore would clobber the phase id.
    if (conversationMode === "deep-plan") return;

    restorePlanningConversationIds({
      conversationId: conversation.id,
      serverPlanningConversationId,
      subConversationTaskId,
      localPlanningConversationId,
      setSubConversationTaskId,
      setLocalPlanningConversationId,
    });
  }, [
    conversation?.id,
    conversationMode,
    serverPlanningConversationId,
    localPlanningConversationId,
    setLocalPlanningConversationId,
    subConversationTaskId,
    setSubConversationTaskId,
  ]);

  const hasCloudPlanner = !!(
    (conversation?.sub_conversation_ids &&
      conversation.sub_conversation_ids.length > 0) ||
    subConversationTaskId
  );
  // Whether a plain `plan`-mode planner helper already exists for this
  // conversation — callers (e.g. the `/plan <task>` interceptor) use this to
  // decide whether they can send a message to the planner immediately, or must
  // wait for creation. Deep Planning resolves its own per-phase planner via
  // `hasPhasePlanner`; a phase planner carries a phase tag, so it must never
  // satisfy the plain-planner check.
  const hasPlanner = isLocalBackend
    ? !!(localPlanningConversationId || serverPlanningConversationId)
    : hasCloudPlanner;

  // Create the local planner for `phase` (or the plain planner when `phase` is
  // null). Records the phase on success so the socket resolves the right one
  // before the tag data refetches.
  const createLocalPlanner = useCallback(
    (
      parentConversationId: string,
      phase: DeepPlanPhaseId | null,
      initialMessage?: string,
    ) => {
      const key = phasePlannerCreationKey(parentConversationId, phase);
      if (inFlightPhasePlannerCreations.has(key)) return;
      inFlightPhasePlannerCreations.add(key);
      createLocalPlanningConversation({
        parentConversationId,
        initialMessage,
        deepPlanPhase: phase,
        deepPlanGuidance: phase ? deepPlanGuidance(t, phase) : null,
      });
    },
    [createLocalPlanningConversation, t],
  );

  const handlePlanClick = useCallback(
    (
      event?: MouseEvent<HTMLButtonElement> | KeyboardEvent,
      initialMessage?: string,
      mode: ConversationMode = "plan",
    ) => {
      event?.preventDefault();
      event?.stopPropagation();

      setConversationMode(mode);

      if (backend.kind !== "cloud") {
        if (!conversation?.id) return;

        // Deep Planning: one planner per phase, each pinned to that phase's
        // document. Create the phase's planner when it does not exist yet; a
        // different phase's planner must not be reused (it edits the wrong
        // file). The in-flight guard stops a double invocation from creating
        // two planners for the same phase.
        if (mode === "deep-plan") {
          const phase = useConversationStore.getState().deepPlan.activePhase;
          // Implementation runs in the code agent, not a planner.
          if (!phase || !isPlanningMode("deep-plan", phase)) return;
          if (hasPhasePlanner || isCreatingLocalPlanningConversation) return;
          createLocalPlanner(conversation.id, phase, initialMessage);
          return;
        }

        // Plain plan mode: one shared, untagged planner.
        if (
          localPlanningConversationId ||
          serverPlanningConversationId ||
          isCreatingLocalPlanningConversation
        ) {
          return;
        }
        createLocalPlanner(conversation.id, null, initialMessage);
        return;
      }

      if (hasCloudPlanner || !conversation?.id) {
        return;
      }

      createConversation(
        {
          parentConversationId: conversation.id,
          agentType: "plan",
          entryPoint: "plan_sub_conversation",
          ...(initialMessage ? { query: initialMessage } : {}),
        },
        {
          onSuccess: (data) => {
            displaySuccessToast(
              t(I18nKey.PLANNING_AGENTT$PLANNING_AGENT_INITIALIZED),
            );
            if (data.task_id) {
              setSubConversationTaskId(data.task_id);
              setConversationState(conversation.id, {
                subConversationTaskId: data.task_id,
              });
            }
          },
        },
      );
    },
    [
      backend.kind,
      conversation,
      createConversation,
      createLocalPlanner,
      createLocalPlanningConversation,
      hasCloudPlanner,
      hasPhasePlanner,
      isCreatingLocalPlanningConversation,
      localPlanningConversationId,
      serverPlanningConversationId,
      setConversationMode,
      setSubConversationTaskId,
      t,
    ],
  );

  // Entering or advancing to a Deep Planning phase must provision that phase's
  // planner: without its own planner pinned to the phase document, messages
  // would run in the previous phase's planner (wrong `plan_path`) or the code
  // agent. Cloud backends keep their existing sub-conversation plumbing.
  const ensureDeepPlanPlanner = useCallback(() => {
    if (backend.kind === "cloud") return;
    if (!conversation?.id) return;
    const phase = useConversationStore.getState().deepPlan.activePhase;
    if (!phase || !isPlanningMode("deep-plan", phase)) return;
    if (hasPhasePlanner || isCreatingLocalPlanningConversation) return;
    createLocalPlanner(conversation.id, phase);
  }, [
    backend.kind,
    conversation?.id,
    createLocalPlanner,
    hasPhasePlanner,
    isCreatingLocalPlanningConversation,
  ]);

  // Provision the active phase's planner as soon as the mode is deep-plan or
  // the phase advances. This hook is mounted by the mode controls, so the
  // planner exists before the user can send into the phase; the guards inside
  // `ensureDeepPlanPlanner` keep concurrent mount points from double-creating.
  useEffect(() => {
    if (conversationMode !== "deep-plan") return;
    ensureDeepPlanPlanner();
  }, [conversationMode, activeDeepPlanPhase, ensureDeepPlanPlanner]);

  return {
    handlePlanClick,
    hasPlanner,
    /** Whether the active Deep Planning phase already has its own planner. */
    hasDeepPlanPlanner: hasPhasePlanner,
    ensureDeepPlanPlanner,
    isCreatingConversation:
      isCreatingCloudConversation || isCreatingLocalPlanningConversation,
  };
};
