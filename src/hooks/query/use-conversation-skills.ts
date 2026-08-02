import { useActiveConversation } from "./use-active-conversation";
import { useSkills } from "./use-skills";

/**
 * Skill metadata used for command discovery and the skills modal. Local
 * conversations query the catalog available in their attached workspace.
 * Conversation-loaded resources belong to `/skills` and deliberately use a
 * separate service owner. "No workspace" falls back to the global working
 * directory.
 */
export const useConversationSkills = () => {
  const conversation = useActiveConversation();
  const projectDir = conversation.data?.selected_workspace ?? undefined;

  return useSkills(projectDir);
};
