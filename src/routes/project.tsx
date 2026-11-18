import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { ArrowLeft, MessageSquare, Pencil } from "lucide-react";
import { HttpError } from "@openhands/typescript-client";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ProjectBoard } from "#/components/features/projects/project-board";
import { ProjectFiles } from "#/components/features/projects/project-files";
import { ProjectMemory } from "#/components/features/projects/project-memory";
import { EditProjectModal } from "#/components/features/projects/edit-project-modal";
import { ProjectsOffNotice } from "#/components/features/projects/projects-off-notice";
import { projectIdKey } from "#/api/projects-service/projects-service.api";
import {
  useProject,
  useProjectConversations,
  useProjectsEnabled,
} from "#/hooks/query/use-projects";
import { useNavigation } from "#/context/navigation-context";
import {
  conversationStateLabel,
  needsUser,
} from "#/utils/project-conversation-state";
import { formControlButtonClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import { PROJECTS_PATH } from "#/utils/project-paths";

/** /projects/:projectId: the coordinator, the board, the files and the memory. */
export default function ProjectPage() {
  const { t } = useTranslation("openhands");
  const { projectId } = useParams();
  const enabled = useProjectsEnabled();
  const { data: project, error } = useProject(projectId);
  const { data: board } = useProjectConversations(projectId);
  const { navigate } = useNavigation();
  const [isEditing, setIsEditing] = React.useState(false);

  const coordinator = board?.coordinator ?? null;
  const conversations = board?.conversations ?? [];
  const titles = React.useMemo(() => {
    const map = new Map<string, string>();
    for (const c of [...(coordinator ? [coordinator] : []), ...conversations]) {
      if (c.title) map.set(projectIdKey(c.conversation_id), c.title);
    }
    return map;
  }, [coordinator, conversations]);
  const coordinatorId =
    coordinator?.conversation_id ?? project?.coordinator_conversation_id;

  let body: React.ReactNode;
  if (!enabled) {
    body = <ProjectsOffNotice />;
  } else if (error instanceof HttpError && error.status === 404) {
    body = (
      <p className="text-sm text-muted">{t(I18nKey.PROJECTS$NOT_FOUND)}</p>
    );
  } else if (!project || !projectId) {
    body = <p className="text-sm text-muted">{t(I18nKey.PROJECTS$LOADING)}</p>;
  } else {
    body = (
      <>
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1
              data-testid="project-name"
              className="text-xl font-medium leading-6 text-foreground"
            >
              {project.name}
            </h1>
            <button
              type="button"
              data-testid="project-edit"
              aria-label={t(I18nKey.PROJECTS$EDIT)}
              title={t(I18nKey.PROJECTS$EDIT)}
              className="rounded p-1 text-muted hover:text-foreground"
              onClick={() => setIsEditing(true)}
            >
              <Pencil width={14} height={14} aria-hidden />
            </button>
          </div>
          {project.description ? (
            <p className="max-w-prose whitespace-pre-wrap text-sm text-muted">
              {project.description}
            </p>
          ) : null}
        </div>

        <section
          data-testid="project-coordinator"
          className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-control bg-base-secondary px-4 py-3"
        >
          <div className="min-w-0 space-y-0.5">
            <h2 className="text-sm font-medium text-foreground">
              {t(I18nKey.PROJECTS$COORDINATOR)}
            </h2>
            <p className="text-xs text-muted">
              {coordinatorId
                ? t(I18nKey.PROJECTS$COORDINATOR_HINT)
                : t(I18nKey.PROJECTS$NO_COORDINATOR)}
            </p>
            {coordinator ? (
              <p
                data-testid="project-coordinator-state"
                data-attention={coordinator.attention}
                className={cn(
                  "text-xs",
                  needsUser(coordinator)
                    ? "font-medium text-foreground"
                    : "text-muted",
                )}
              >
                {t(conversationStateLabel(coordinator))}
                {coordinator.turn === "asking" && coordinator.ask
                  ? `: ${coordinator.ask}`
                  : null}
              </p>
            ) : null}
          </div>
          {coordinatorId ? (
            <NavigationLink
              to={`/conversations/${projectIdKey(coordinatorId)}`}
              data-testid="project-open-coordinator"
              className={cn(
                formControlButtonClassName,
                "flex items-center gap-2 bg-primary text-[var(--oh-accent-foreground)] hover:opacity-80",
              )}
            >
              <MessageSquare width={16} height={16} aria-hidden />
              {t(I18nKey.PROJECTS$OPEN_COORDINATOR)}
            </NavigationLink>
          ) : null}
        </section>

        <section
          aria-labelledby="project-board-title"
          className="flex flex-col gap-3"
        >
          <h2
            id="project-board-title"
            className="text-base font-medium text-foreground"
          >
            {t(I18nKey.PROJECTS$CONVERSATIONS)}
          </h2>
          <ProjectBoard projectId={projectId} conversations={conversations} />
        </section>

        <ProjectFiles projectId={projectId} titles={titles} />
        <ProjectMemory projectId={projectId} />
        {isEditing ? (
          <EditProjectModal
            project={project}
            onClose={() => setIsEditing(false)}
            onDeleted={() => navigate(PROJECTS_PATH)}
          />
        ) : null}
      </>
    );
  }

  return (
    <div data-testid="project-page" className="flex h-full">
      <main className={settingsLikeMainScrollClassName}>
        <div className="mx-auto flex w-full min-w-0 max-w-260 flex-col gap-6">
          <NavigationLink
            to={PROJECTS_PATH}
            end
            className="inline-flex w-fit items-center gap-1 text-sm text-muted hover:text-foreground"
          >
            <ArrowLeft width={14} height={14} aria-hidden />
            {t(I18nKey.PROJECTS$ALL_PROJECTS)}
          </NavigationLink>
          {body}
        </div>
      </main>
    </div>
  );
}
