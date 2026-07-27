import React from "react";
import { useNavigate, useLocation, useMatch } from "react-router";
import { useTranslation } from "react-i18next";

import { useConversationId } from "#/hooks/use-conversation-id";
import { useCommandStore } from "#/stores/command-store";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentStore } from "#/stores/agent-store";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  clearLastConversationId,
  setLastConversationId,
} from "#/api/backend-registry/last-conversation-store";
import { AgentState } from "#/types/agent-state";

import { EventHandler } from "../wrapper/event-handler";

import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useTaskPollingController } from "#/hooks/query/use-task-polling";

import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { ConversationMain } from "#/components/features/conversation/conversation-main/conversation-main";
import { ConversationMobilePanelPage } from "#/components/features/conversation/conversation-main/conversation-mobile-panel-page";

import { WebSocketProviderWrapper } from "#/contexts/websocket-provider-wrapper";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { I18nKey } from "#/i18n/declaration";
import { resumeCloudSandbox } from "#/api/cloud/conversation-service.api";

function AppContent() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const panelViewMatch = useMatch("/conversations/:conversationId/panel");

  const { isTask, taskStatus, taskDetail } = useTaskPollingController();

  // The conversationId in the URL belongs to whichever backend was
  // active when the route first mounted. If the user switches backends
  // while this route is still mounted, the id is meaningless under the
  // new backend — disable the active-conversation fetch (and its 404
  // toast) so we don't fire a request that the BackendSelector's
  // redirect will immediately navigate away from anyway. Mirrors the
  // same guard in `routes/automation-detail.tsx`.
  const active = useActiveBackend();
  const mountedBackendId = React.useRef(active.backend.id);
  const mountedOrgId = React.useRef(active.orgId);
  const backendChanged =
    mountedBackendId.current !== active.backend.id ||
    mountedOrgId.current !== active.orgId;

  const { data: conversation, isFetched } = useActiveConversation();
  const { data: isAuthed } = useIsAuthed();
  const { resetConversationState } = useConversationStore();
  const navigate = useNavigate();
  const location = useLocation();
  const clearTerminal = useCommandStore((state) => state.clearTerminal);
  const resetConversationRuntimeState = useConversationStateStore(
    (state) => state.reset,
  );
  const setCurrentAgentState = useAgentStore(
    (state) => state.setCurrentAgentState,
  );
  const removeErrorMessage = useErrorMessageStore(
    (state) => state.removeErrorMessage,
  );

  // Per-conversation UI/runtime resets. The event store is cleared separately,
  // inside ConversationWebSocketProvider, so the clear is ordered *before* the
  // preloaded-history re-seed (see the note there) — clearing it here would run
  // too late and wipe the freshly seeded history on a conversation switch.
  React.useEffect(() => {
    clearTerminal();
    resetConversationState();
    resetConversationRuntimeState();
    setCurrentAgentState(AgentState.LOADING);
    removeErrorMessage();
  }, [
    conversationId,
    clearTerminal,
    resetConversationState,
    resetConversationRuntimeState,
    setCurrentAgentState,
    removeErrorMessage,
  ]);

  React.useEffect(() => {
    if (isTask && taskStatus === "ERROR") {
      displayErrorToast(
        taskDetail || t(I18nKey.CONVERSATION$FAILED_TO_START_FROM_TASK),
      );
      // Navigate back to the original conversation when a resume task fails so
      // the user isn't stranded at the dead task-{id} URL. The resume effect's
      // ref prevents it from immediately retrying once we land there.
      const resumedFrom = (location.state as Record<string, unknown> | null)
        ?.resumedFromConversationId as string | undefined;
      navigate(
        resumedFrom ? `/conversations/${resumedFrom}` : "/conversations",
        { replace: true },
      );
    }
  }, [isTask, taskStatus, taskDetail, t, navigate, location.state]);

  React.useEffect(() => {
    if (!isFetched || !isAuthed) return;
    // The BackendSelector is in the middle of redirecting us away from
    // this route — don't toast/navigate based on a 404 that's just
    // "this id doesn't exist on the new backend".
    if (backendChanged) return;

    if (!conversation) {
      // Clear the per-backend "last selected" slot so the next switch
      // to this backend doesn't try to revisit a stale id.
      clearLastConversationId(active.backend.id, active.orgId);
      displayErrorToast(t(I18nKey.CONVERSATION$NOT_EXIST_OR_NO_PERMISSION));
      navigate("/conversations");
    }
  }, [
    conversation,
    isFetched,
    isAuthed,
    navigate,
    t,
    backendChanged,
    active.backend.id,
    active.orgId,
  ]);

  // Remember the most recently selected conversation for the current
  // (backend, org) so flipping back to this backend later restores the
  // user to where they left off. Skip while a backend switch is in
  // flight: the id in the URL is from the previous backend and would
  // otherwise overwrite the new backend's memory.
  React.useEffect(() => {
    if (backendChanged) return;
    if (!conversationId) return;
    if (conversationId.startsWith("task-")) return;
    setLastConversationId(active.backend.id, active.orgId, conversationId);
  }, [conversationId, backendChanged, active.backend.id, active.orgId]);

  // Cloud conversation resume: mirrors OpenHands' useSandboxRecovery.
  //
  // When the cloud API reports sandbox_status === "PAUSED" the sandbox is
  // sleeping. The correct wake-up call is POST /api/v1/sandboxes/{id}/resume
  // (a lightweight unpause). The previous approach — creating a new start task
  // via POST /api/v1/app-conversations — was wrong: it tries to provision a
  // fresh conversation in the sandbox and is subject to a 120-second cold-start
  // timeout that can fail. The resume endpoint simply unpauses the existing one.
  //
  // After calling resume we stay on the current URL. The 3-second refetch
  // interval in useActiveConversation (active while conversation_url is null)
  // polls until conversation_url populates, then the WebSocket connects.
  //
  // A ref guards against duplicate triggers per unique conversation.id within
  // the same route-mount lifetime.
  const resumeTriggeredForRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!isFetched || !conversation) return;
    if (active.backend.kind !== "cloud") return;
    if (conversation.sandbox_status !== "PAUSED") return; // only resume PAUSED sandboxes
    if (!conversation.sandbox_id) return; // no sandbox to resume
    if (resumeTriggeredForRef.current === conversation.id) return; // already sent

    resumeTriggeredForRef.current = conversation.id;

    resumeCloudSandbox(conversation.sandbox_id).catch(() => {
      displayErrorToast(t(I18nKey.CONVERSATION$FAILED_TO_START_FROM_TASK));
    });
  }, [
    isFetched,
    conversation?.id,
    conversation?.sandbox_status,
    conversation?.sandbox_id,
    active.backend.kind,
    t,
  ]);

  // A backend switch is in flight (BackendSelector flips the active backend
  // and redirects to /conversations on the next tick). The conversationId in
  // the URL belongs to the *previous* backend, so unmount the whole
  // conversation subtree now — before any per-conversation query (history,
  // metrics, sub-conversations, runtime info, …) re-fires against a backend
  // the id is foreign to. Those foreign fetches fail response validation and
  // surface "agent server returned data this UI does not understand". This is
  // deterministic regardless of the navigate-vs-setActive render race: React
  // re-renders this parent before its children, so returning null removes them
  // before they can issue the request.
  if (backendChanged) {
    return null;
  }

  const content = (
    <EventHandler>
      <div data-testid="app-route" className="flex h-full flex-col">
        {panelViewMatch ? (
          <ConversationMobilePanelPage
            onNavigateBack={() => navigate(`/conversations/${conversationId}`)}
          />
        ) : (
          <ConversationMain />
        )}
      </div>
    </EventHandler>
  );

  return (
    <WebSocketProviderWrapper conversationId={conversationId}>
      {content}
    </WebSocketProviderWrapper>
  );
}

export function ConversationView() {
  return <AppContent />;
}

export default ConversationView;
