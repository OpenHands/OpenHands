import { useActiveConversation } from "./use-active-conversation";
import { useSkills } from "./use-skills";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";

/**
 * Skill metadata used for command discovery and the skills modal. Local
 * conversations query the catalog available in their attached workspace.
 * Conversation-loaded resources belong to `/skills` and deliberately use a
 * separate service owner. "No workspace" falls back to the global working
 * directory.
 */
export const useConversationSkills = () => {
  const { conversationId: routeConversationId } = useOptionalConversationId();
  const conversation = useActiveConversation();
  const hasConversationRoute = !!routeConversationId;
  const hasResolvedConversation =
    !!routeConversationId && conversation.data?.id === routeConversationId;
  const projectDir = hasResolvedConversation
    ? (conversation.data?.selected_workspace ?? undefined)
    : undefined;

  // A task route and a real conversation route whose metadata has not loaded
  // yet have no authoritative workspace. Querying with `undefined` in those
  // states would silently substitute the global catalog and briefly advertise
  // commands from the wrong workspace.
  return useSkills(projectDir, {
    enabled: !hasConversationRoute || hasResolvedConversation,
    // A disabled query can still expose cached data. Give unresolved routes a
    // distinct key so a warm home/global catalog cannot leak into /help or
    // autocomplete while conversation metadata is still loading.
    ...(!hasConversationRoute || hasResolvedConversation
      ? {}
      : { queryScope: `unresolved:${routeConversationId}` }),
  });
};
