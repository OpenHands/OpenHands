import { useMemo } from "react";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";

/**
 * Plugins loaded into the active conversation, read from the client-side
 * metadata snapshot taken at creation (explicitly attached plugins plus the
 * enabled installed plugins the SDK auto-loads). Empty when none are loaded or
 * when used outside a conversation route. The agent-server doesn't return a
 * live conversation's loaded plugins, so this is the available source today.
 */
export function useConversationPlugins(): PluginSpec[] {
  const { conversationId } = useOptionalConversationId();
  return useMemo(() => {
    if (!conversationId) return [];
    return getStoredConversationMetadata(conversationId)?.plugins ?? [];
  }, [conversationId]);
}
