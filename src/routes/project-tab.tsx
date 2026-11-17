import { ProjectPanel } from "#/components/features/projects/project-panel/project-panel";
import { useConversationProjectId } from "#/hooks/query/use-conversation-project";
import { useConversationId } from "#/hooks/use-conversation-id";

/** The conversation's project, beside the chat. Shown only for project members. */
function ProjectTab() {
  const projectId = useConversationProjectId();
  const { conversationId } = useConversationId();
  if (!projectId) return null;
  return <ProjectPanel projectId={projectId} conversationId={conversationId} />;
}

export default ProjectTab;
