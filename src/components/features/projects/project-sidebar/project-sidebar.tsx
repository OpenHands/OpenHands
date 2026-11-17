import { useTranslation } from "react-i18next";
import { FolderKanban } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useProject } from "#/hooks/query/use-projects";
import { projectPath } from "#/utils/project-paths";
import { cn } from "#/utils/utils";
import { ProjectConversationList } from "./project-conversation-list";

/**
 * The sidebar in a project view: the project's name, which opens its page,
 * and its conversations in place of the list of all conversations.
 */
export function ProjectSidebar({
  projectId,
  conversationId,
  onProjectPage,
}: {
  projectId: string;
  /** The open conversation, if it belongs to the project. */
  conversationId?: string;
  /** Whether the project's own page is open. */
  onProjectPage: boolean;
}) {
  const { t } = useTranslation("openhands");
  const { data: project } = useProject(projectId);
  return (
    <div
      data-testid="project-sidebar"
      className="flex min-h-0 w-full flex-1 flex-col"
    >
      <NavigationLink
        to={projectPath(projectId)}
        data-testid="project-sidebar-name"
        aria-current={onProjectPage ? "page" : undefined}
        className={cn(
          "mr-2.5 flex min-w-0 items-center gap-2 rounded-lg px-2 py-2 text-sm font-medium",
          "text-muted transition-colors hover:text-contrast",
          onProjectPage && "bg-surface text-foreground",
        )}
      >
        <FolderKanban width={16} height={16} aria-hidden className="shrink-0" />
        <span className="truncate">
          {project?.name ?? t(I18nKey.PROJECTS$LOADING)}
        </span>
      </NavigationLink>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-3 pr-2.5 pt-2 custom-scrollbar-always">
        <ProjectConversationList
          projectId={projectId}
          conversationId={conversationId}
        />
      </div>
    </div>
  );
}
