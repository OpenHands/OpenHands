import { useCallback, useMemo } from "react";
import { useEventStore } from "#/stores/use-event-store";
import { useSkillInstallBannerStore } from "#/stores/skill-install-banner-store";
import { detectSkillInstalls } from "#/utils/skill-install-events";

/**
 * Skill installs performed by the agent in this conversation (via the
 * bundled add-skill flow), minus the ones the user dismissed. The event
 * store is global, so results are scoped to the conversation it currently
 * holds — a mismatched id (remount race) yields no installs.
 */
export const useSkillInstalls = (conversationId: string | null | undefined) => {
  const events = useEventStore((s) => s.events);
  const loadedConversationId = useEventStore((s) => s.loadedConversationId);
  const dismissedEventIds = useSkillInstallBannerStore(
    (s) => s.dismissedEventIds,
  );
  const dismiss = useSkillInstallBannerStore((s) => s.dismiss);

  const installs = useMemo(() => {
    if (!conversationId || conversationId !== loadedConversationId) return [];
    return detectSkillInstalls(events).filter(
      (install) => !dismissedEventIds[install.eventId],
    );
  }, [conversationId, loadedConversationId, events, dismissedEventIds]);

  const dismissAll = useCallback(
    () => dismiss(installs.map((install) => install.eventId)),
    [dismiss, installs],
  );

  return { installs, dismissAll };
};
