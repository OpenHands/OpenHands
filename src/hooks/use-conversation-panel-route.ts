import React from "react";
import { useMatch, useNavigate } from "react-router";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { subscribeToPanelReveal } from "#/services/canvas-ui";
import { useConversationStore } from "#/stores/conversation-store";

/**
 * Keeps the conversation drawer's route in step with the viewport. At phone
 * width the drawer is the full-screen `/conversations/:id/panel` page; above
 * the breakpoint it sits beside the chat on `/conversations/:id`.
 *
 * Returns whether to render the full-screen mobile panel page.
 */
export function useConversationPanelRoute(conversationId: string): boolean {
  const isMobile = useBreakpoint();
  const isPanelRoute = Boolean(
    useMatch("/conversations/:conversationId/panel"),
  );
  const navigate = useNavigate();

  // Agent `open_tab` / `navigate_to_file` and chat file links reveal the
  // drawer through canvas-ui. At phone width that means opening the panel page.
  React.useEffect(() => {
    if (!isMobile || isPanelRoute || !conversationId) return undefined;
    return subscribeToPanelReveal(() =>
      navigate(`/conversations/${conversationId}/panel`),
    );
  }, [isMobile, isPanelRoute, conversationId, navigate]);

  // A panel page widened past the breakpoint (or a /panel link opened on
  // desktop) falls back to the split chat | drawer layout with the drawer open.
  // This runs after the panel page's unmount cleanup has closed the drawer.
  React.useEffect(() => {
    if (isMobile || !isPanelRoute || !conversationId) return;
    const { setHasRightPanelToggled, setIsRightPanelShown } =
      useConversationStore.getState();
    setHasRightPanelToggled(true);
    setIsRightPanelShown(true);
    navigate(`/conversations/${conversationId}`, { replace: true });
  }, [isMobile, isPanelRoute, conversationId, navigate]);

  return isPanelRoute && isMobile;
}
