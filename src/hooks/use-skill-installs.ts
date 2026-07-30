import { useCallback, useMemo } from "react";
import {
  useConversationEvents,
  useIsConversationLoaded,
} from "#/hooks/use-conversation-events";
import { useSkillInstallBannerStore } from "#/stores/skill-install-banner-store";
import { detectSkillInstalls } from "#/utils/skill-install-events";

/**
 * Skill installs performed by the agent in this conversation (via the
 * bundled add-skill flow), minus the ones the user dismissed. Results are
 * scoped to the conversation bucket currently in navigation scope — a
 * mismatched id (remount race before the bucket is loaded) yields no installs.
 */
export const useSkillInstalls = (conversationId: string | null | undefined) => {
  const events = useConversationEvents(conversationId);
  const isLoaded = useIsConversationLoaded(conversationId);
  const dismissedEventIds = useSkillInstallBannerStore(
    (s) => s.dismissedEventIds,
  );
  const dismiss = useSkillInstallBannerStore((s) => s.dismiss);

  const installs = useMemo(() => {
    if (!conversationId || !isLoaded) return [];
    return detectSkillInstalls(events).filter(
      (install) => !dismissedEventIds[install.eventId],
    );
  }, [conversationId, isLoaded, events, dismissedEventIds]);

  const dismissAll = useCallback(
    () => dismiss(installs.map((install) => install.eventId)),
    [dismiss, installs],
  );

  return { installs, dismissAll };
};
