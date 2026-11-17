import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  useProjectArtifacts,
  useProjectConversations,
} from "#/hooks/query/use-projects";
import { pullRequestRef } from "#/hooks/query/use-pull-request-status";
import { cn } from "#/utils/utils";
import { ProjectPanelFiles } from "./project-panel-files";
import { ProjectPanelPullRequests } from "./project-panel-prs";

type ProjectPanelView = "files" | "prs";

/**
 * What the project made, beside a conversation that belongs to it: the files
 * its conversations saved and the pull requests they opened. Its
 * conversations are in the sidebar; managing the project (resolve, remove,
 * delete files, memory) stays on the project page.
 */
export function ProjectPanel({ projectId }: { projectId: string }) {
  const { t } = useTranslation("openhands");
  const { data: artifacts } = useProjectArtifacts(projectId);
  const { data: board } = useProjectConversations(projectId);
  // Two conversations can name the same pull request; count it once.
  const pullRequestCount = new Set(
    [board?.coordinator, ...(board?.conversations ?? [])].flatMap(
      (c) => c?.pull_requests.map(pullRequestRef) ?? [],
    ),
  ).size;
  const counted = (label: string, count: number | undefined) =>
    count ? `${label} · ${count}` : label;
  const [view, setView] = React.useState<ProjectPanelView>("files");

  const views: Array<{ id: ProjectPanelView; label: string }> = [
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

      {/* Both views stay mounted, so folded groups and an open preview
          survive a switch to the other view and back. */}
      <div
        id="project-panel-view"
        role="tabpanel"
        aria-labelledby={`project-panel-tab-${view}`}
        className="min-h-0"
      >
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
