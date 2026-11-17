import { ProjectPanel } from "#/components/features/projects/project-panel/project-panel";
import { useConversationProjectId } from "#/hooks/query/use-conversation-project";

/** The files and pull requests of the conversation's project, beside the chat. */
function ProjectTab() {
  const projectId = useConversationProjectId();
  if (!projectId) return null;
  return <ProjectPanel projectId={projectId} />;
}

export default ProjectTab;
