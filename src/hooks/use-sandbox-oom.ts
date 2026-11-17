import { useCallback, useMemo } from "react";
import { useEventStore } from "#/stores/use-event-store";
import { useSandboxOOMStore } from "#/stores/sandbox-oom-store";
import {
  detectSandboxOOM,
  type DetectedSandboxOOM,
} from "#/utils/sandbox-oom-events";

/**
 * Tracks Out-Of-Memory (OOM) interruptions in the current conversation,
 * minus the ones the user has explicitly dismissed.
 */
export const useSandboxOOM = (conversationId: string | null | undefined) => {
  const events = useEventStore((s) => s.events);
  const loadedConversationId = useEventStore((s) => s.loadedConversationId);
  const dismissedEventIds = useSandboxOOMStore((s) => s.dismissedEventIds);
  const dismiss = useSandboxOOMStore((s) => s.dismiss);

  const activeOOM = useMemo<DetectedSandboxOOM | null>(() => {
    if (!conversationId || conversationId !== loadedConversationId) return null;
    const detected = detectSandboxOOM(events);
    const unacknowledged = detected.filter(
      (oom) => !dismissedEventIds[oom.eventId],
    );
    return unacknowledged.at(-1) ?? null;
  }, [conversationId, loadedConversationId, events, dismissedEventIds]);

  const dismissActiveOOM = useCallback(() => {
    if (activeOOM) {
      dismiss([activeOOM.eventId]);
    }
  }, [activeOOM, dismiss]);

  return { activeOOM, dismissActiveOOM };
};
