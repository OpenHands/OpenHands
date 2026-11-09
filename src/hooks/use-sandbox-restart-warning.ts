import React from "react";
import { useTranslation } from "react-i18next";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useMainWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

const UNKNOWN_RESTART_REASON = "Unknown";

/**
 * Warns the user when the active Cloud conversation's sandbox has restarted
 * (for example because it was OOM-killed), which stops any processes running
 * in it.
 *
 * A restart is detected by the sandbox's restart count rising between reads.
 * A count that drops means the pod was replaced, which starts a new baseline,
 * and an unknown count (null) leaves the baseline alone.
 *
 * A dropped WebSocket triggers an immediate refetch so a restart is noticed
 * straight away rather than on the next poll.
 */
export function useSandboxRestartWarning() {
  const { t } = useTranslation("openhands");
  const { data: conversation, refetch } = useActiveConversation();
  const connectionState = useMainWebSocketStatus();

  const lastSeen = React.useRef<{ id: string; count: number } | null>(null);
  const previousConnectionState = React.useRef(connectionState);

  React.useEffect(() => {
    if (
      previousConnectionState.current === "OPEN" &&
      connectionState === "CLOSED"
    ) {
      refetch();
    }
    previousConnectionState.current = connectionState;
  }, [connectionState, refetch]);

  const conversationId = conversation?.id;
  const restartCount = conversation?.sandbox_restart_count;
  const restartReason = conversation?.sandbox_restart_reasons?.join(", ");

  React.useEffect(() => {
    if (conversationId === undefined) return;
    if (restartCount === null || restartCount === undefined) return;

    const previous = lastSeen.current;
    lastSeen.current = { id: conversationId, count: restartCount };

    if (previous?.id === conversationId && restartCount > previous.count) {
      displayErrorToast(
        t(I18nKey.CONVERSATION$SANDBOX_RESTARTED, {
          reason: restartReason || UNKNOWN_RESTART_REASON,
        }),
      );
    }
  }, [conversationId, restartCount, restartReason, t]);
}
