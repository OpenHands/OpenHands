import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, MessageSquare, Pencil, Plus } from "lucide-react";
import { HttpError } from "@openhands/typescript-client";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ProjectBoard } from "#/components/features/projects/project-board";
import { ProjectFiles } from "#/components/features/projects/project-files";
import { ProjectMemory } from "#/components/features/projects/project-memory";
import { EditProjectModal } from "#/components/features/projects/edit-project-modal";
import { ProjectsOffNotice } from "#/components/features/projects/projects-off-notice";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import ProjectsService, {
  projectIdKey,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  projectsQueryRoot,
  useProject,
  useProjectConversations,
  useProjectsEnabled,
} from "#/hooks/query/use-projects";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useNavigation } from "#/context/navigation-context";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import {
  conversationStateLabel,
  needsUser,
} from "#/utils/project-conversation-state";
import {
  formControlButtonClassName,
  formControlFieldClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import { PROJECTS_PATH } from "#/utils/project-paths";

/** Pick a recent conversation that is not in this project yet, and add it. */
function AddConversation({
  projectId,
  memberIds,
}: {
  projectId: string;
  memberIds: Set<string>;
}) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const [selected, setSelected] = React.useState("");
  const { data } = useQuery({
    queryKey: [...projectsQueryRoot(backend), "recent-conversations"],
    queryFn: () => AgentServerConversationService.searchConversations(50),
    staleTime: 30_000,
    meta: { disableToast: true },
  });
  const candidates = (data?.items ?? []).filter(
    (c) => !memberIds.has(projectIdKey(c.id)),
  );
  const add = useMutation({
    mutationFn: () => ProjectsService.addConversation(projectId, selected),
    meta: { disableToast: true },
    onSuccess: () => setSelected(""),
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) }),
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPDATE_FAILED)),
  });
  if (!candidates.length) return null;
  return (
    <form
      className="flex items-center gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (selected) add.mutate();
      }}
    >
      <select
        data-testid="project-add-conversation"
        aria-label={t(I18nKey.PROJECTS$ADD_CONVERSATION)}
        className={`${formControlFieldClassName} max-w-80`}
        value={selected}
        onChange={(event) => setSelected(event.target.value)}
      >
        <option value="">{t(I18nKey.PROJECTS$ADD_CONVERSATION)}</option>
        {candidates.map((c) => (
          <option key={c.id} value={c.id}>
            {c.title || c.id}
          </option>
        ))}
      </select>
      <BrandButton
        type="submit"
        variant="secondary"
        isDisabled={!selected || add.isPending}
      >
        {t(I18nKey.BUTTON$ADD)}
      </BrandButton>
    </form>
  );
}

/** Start a conversation that belongs to this project from its first message. */
function NewConversation({ projectId }: { projectId: string }) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { navigate } = useNavigation();
  const { mutateAsync: createConversation } = useCreateConversation({
    disableToast: true,
  });
  const start = useMutation({
    mutationFn: async () => {
      const created = await createConversation({ entryPoint: "project_page" });
      await ProjectsService.addConversation(projectId, created.conversation_id);
      return created.conversation_id;
    },
    meta: { disableToast: true },
    onSuccess: (conversationId) => {
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) });
      navigate(`/conversations/${conversationId}`);
    },
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPDATE_FAILED)),
  });
  return (
    <BrandButton
      testId="project-new-conversation"
      type="button"
      variant="secondary"
      isDisabled={start.isPending}
      aria-busy={start.isPending}
      startContent={<Plus width={14} height={14} />}
      onClick={() => start.mutate()}
    >
      {t(I18nKey.PROJECTS$NEW_CONVERSATION)}
    </BrandButton>
  );
}

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
  const memberIds = React.useMemo(
    () =>
      new Set(
        [...(coordinator ? [coordinator] : []), ...conversations].map((c) =>
          projectIdKey(c.conversation_id),
        ),
      ),
    [coordinator, conversations],
  );
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
                {coordinator.turn === "asked" && coordinator.ask
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
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2
              id="project-board-title"
              className="text-base font-medium text-foreground"
            >
              {t(I18nKey.PROJECTS$CONVERSATIONS)}
            </h2>
            <div className="flex flex-wrap items-center gap-2">
              <AddConversation projectId={projectId} memberIds={memberIds} />
              <NewConversation projectId={projectId} />
            </div>
          </div>
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
