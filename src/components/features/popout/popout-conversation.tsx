import React from "react";
import {
  NavigationProvider,
  useNavigation,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { ConversationRenderScopeProvider } from "#/contexts/conversation-render-scope";
import { WebSocketProviderWrapper } from "#/contexts/websocket-provider-wrapper";
import { EventHandler } from "#/wrapper/event-handler";
import { ChatInterface } from "#/components/features/chat/chat-interface";

interface PopoutConversationProps {
  conversationId: string;
}

/**
 * Live chat for a popped-out conversation, scoped so it coexists with the
 * primary routed conversation: nested navigation supplies the id, the
 * render-scope
 * flag keeps process-wide singletons alone, and the WS provider writes only
 * into this conversation's event-store bucket.
 */
export function PopoutConversation({
  conversationId,
}: PopoutConversationProps) {
  const outerNavigation = useNavigation();

  const navigationValue = React.useMemo<NavigationContextValue>(
    () => ({
      currentPath: `/conversations/${conversationId}`,
      conversationId,
      // Popouts don't own the URL — forward navigations (e.g. maximize) to the
      // outer router so the primary route updates.
      isNavigating: outerNavigation.isNavigating,
      navigate: outerNavigation.navigate,
    }),
    [conversationId, outerNavigation.isNavigating, outerNavigation.navigate],
  );

  return (
    <ConversationRenderScopeProvider isPrimary={false}>
      <NavigationProvider value={navigationValue}>
        <WebSocketProviderWrapper
          conversationId={conversationId}
          sharedSideEffects={false}
        >
          <EventHandler>
            <div
              data-testid="popout-conversation"
              className="flex h-full min-h-0 w-full flex-col"
            >
              <ChatInterface />
            </div>
          </EventHandler>
        </WebSocketProviderWrapper>
      </NavigationProvider>
    </ConversationRenderScopeProvider>
  );
}
