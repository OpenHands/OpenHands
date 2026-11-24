import { useTranslation } from "react-i18next";
import { ChevronRight } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useProject } from "#/hooks/query/use-projects";
import { projectPath } from "#/utils/project-paths";
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
  const name = project?.name ?? t(I18nKey.PROJECTS$LOADING);
  return (
    <div
      data-testid="project-sidebar"
      className="flex min-h-0 w-full flex-1 flex-col"
    >
      {/* A break from the app's navigation above, then the project's name
          as the title of the list under it, set like the project page's. */}
      <header className="-ml-2.5 mt-2 w-[calc(100%+0.625rem)] border-t border-border pb-1 pl-4 pr-2.5 pt-3">
        <p className="text-2xs font-medium uppercase tracking-wide text-muted">
          {t(I18nKey.PROJECTS$PROJECT_TAB)}
        </p>
        <h2 className="min-w-0 text-base font-medium leading-6 text-foreground">
          {onProjectPage ? (
            <span
              data-testid="project-sidebar-name"
              aria-current="page"
              className="block truncate"
            >
              {name}
            </span>
          ) : (
            <NavigationLink
              to={projectPath(projectId)}
              data-testid="project-sidebar-name"
              title={t(I18nKey.PROJECTS$OPEN_PROJECT_PAGE)}
              className="group inline-flex max-w-full items-center gap-1 hover:underline"
            >
              <span className="truncate">{name}</span>
              <ChevronRight
                width={14}
                height={14}
                aria-hidden
                className="shrink-0 text-muted group-hover:text-foreground"
              />
            </NavigationLink>
          )}
        </h2>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain pb-3 pl-1.5 pr-2.5 pt-1 custom-scrollbar-always">
        <ProjectConversationList
          projectId={projectId}
          conversationId={conversationId}
        />
      </div>
    </div>
  );
}
