import { useActiveConversation } from "./use-active-conversation";
import { useSkills } from "./use-skills";

/**
 * Skills catalog scoped to the active conversation's attached workspace, so
 * the slash-command menu and skills modal list the same project skills that
 * were loaded into the conversation. Falls back to the global workspace dir
 * for "No workspace" conversations (``selected_workspace`` is null).
 */
export const useConversationSkills = () => {
  const conversation = useActiveConversation();
  return useSkills(conversation.data?.selected_workspace ?? undefined);
};
