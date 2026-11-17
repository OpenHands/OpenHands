import React from "react";
import { useTranslation } from "react-i18next";
import { ChevronRight } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import {
  useProject,
  useProjectArtifacts,
  useProjectConversations,
} from "#/hooks/query/use-projects";
import { projectPath } from "#/utils/project-paths";
import { cn } from "#/utils/utils";
import { ProjectPanelConversations } from "./project-panel-conversations";
import { ProjectPanelFiles } from "./project-panel-files";
import { ProjectPanelPullRequests } from "./project-panel-prs";

type ProjectPanelView = "conversations" | "files" | "prs";

/**
 * The project at a glance, beside a conversation that belongs to it: what
 * its conversations are doing, the files they made and the pull requests
 * they opened. Managing the project
 * (resolve, remove, delete files, memory) stays on the project page.
 */
export function ProjectPanel({
  projectId,
  conversationId,
}: {
  projectId: string;
  /** The open conversation, highlighted in the list. */
  conversationId: string;
}) {
  const { t } = useTranslation("openhands");
  const { data: project } = useProject(projectId);
  const { data: artifacts } = useProjectArtifacts(projectId);
  const { data: board } = useProjectConversations(projectId);
  const pullRequestCount = board
    ? [board.coordinator, ...board.conversations].reduce(
        (sum, c) => sum + (c?.pull_requests.length ?? 0),
        0,
      )
    : 0;
  const counted = (label: string, count: number | undefined) =>
    count ? `${label} · ${count}` : label;
  const [view, setView] = React.useState<ProjectPanelView>("conversations");

  const views: Array<{ id: ProjectPanelView; label: string }> = [
    { id: "conversations", label: t(I18nKey.PROJECTS$CONVERSATIONS) },
    {
      id: "files",
      label: counted(t(I18nKey.PROJECTS$FILES), artifacts?.length),
    },
    { id: "prs", label: counted(t(I18nKey.PROJECTS$PRS), pullRequestCount) },
  ];

  return (
    <div
      data-testid="project-panel"
      className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto px-4 py-3"
    >
      <div className="flex items-baseline justify-between gap-3">
        <h2
          data-testid="project-panel-name"
          className="min-w-0 truncate text-base font-medium text-foreground"
        >
          {project?.name ?? t(I18nKey.PROJECTS$LOADING)}
        </h2>
        <NavigationLink
          to={projectPath(projectId)}
          data-testid="project-panel-open-project"
          className="inline-flex shrink-0 items-center gap-0.5 text-xs text-muted hover:text-foreground"
        >
          {t(I18nKey.PROJECTS$OPEN_PROJECT_PAGE)}
          <ChevronRight width={12} height={12} aria-hidden />
        </NavigationLink>
      </div>

      <div
        role="tablist"
        aria-label={t(I18nKey.PROJECTS$PROJECT_TAB)}
        className="flex rounded-full bg-surface p-1"
      >
        {views.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`project-panel-tab-${id}`}
            aria-selected={view === id}
            aria-controls="project-panel-view"
            data-testid={`project-panel-tab-${id}`}
            className={cn(
              "flex-1 rounded-full px-3 py-1 text-sm transition-colors",
              view === id
                ? "bg-surface-raised text-foreground shadow-sm"
                : "text-muted hover:text-foreground",
            )}
            onClick={() => setView(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Every view stays mounted, so its folded groups and open preview
          survive a switch to another view and back. */}
      <div
        id="project-panel-view"
        role="tabpanel"
        aria-labelledby={`project-panel-tab-${view}`}
        className="min-h-0"
      >
        <div hidden={view !== "conversations"}>
          <ProjectPanelConversations
            projectId={projectId}
            conversationId={conversationId}
          />
        </div>
        <div hidden={view !== "files"}>
          <ProjectPanelFiles projectId={projectId} />
        </div>
        <div hidden={view !== "prs"}>
          <ProjectPanelPullRequests projectId={projectId} />
        </div>
      </div>
    </div>
  );
}
