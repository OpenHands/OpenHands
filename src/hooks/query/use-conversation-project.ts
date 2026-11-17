import { useActiveConversation } from "./use-active-conversation";
import { useProjectsEnabled } from "./use-projects";

/**
 * The project the open conversation belongs to (its `project` tag), or
 * undefined when it belongs to none or Projects is turned off.
 */
export function useConversationProjectId(): string | undefined {
  const enabled = useProjectsEnabled();
  const { data: conversation } = useActiveConversation();
  return enabled ? (conversation?.tags?.project ?? undefined) : undefined;
}
