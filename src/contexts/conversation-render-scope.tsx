import React from "react";

/**
 * Distinguishes the primary routed conversation from a secondary one rendered
 * alongside it (a popout). Secondary scopes must not claim process-wide
 * singletons (ConversationService.currentConversation, shared chrome stores).
 */
export interface ConversationRenderScope {
  isPrimary: boolean;
}

const ConversationRenderScopeContext =
  React.createContext<ConversationRenderScope>({ isPrimary: true });

export function ConversationRenderScopeProvider({
  isPrimary,
  children,
}: {
  isPrimary: boolean;
  children: React.ReactNode;
}) {
  const value = React.useMemo(() => ({ isPrimary }), [isPrimary]);
  return (
    <ConversationRenderScopeContext.Provider value={value}>
      {children}
    </ConversationRenderScopeContext.Provider>
  );
}

export function useConversationRenderScope(): ConversationRenderScope {
  return React.useContext(ConversationRenderScopeContext);
}
