import React from "react";
import { useTranslation } from "react-i18next";
import {
  PROJECTS_PATH,
  projectDetailPath,
  projectIdFromPath,
} from "#/api/projects-service/projects-constants";
import { CreateProjectModal } from "#/components/features/projects/create-project-modal";
import { ProjectCard } from "#/components/features/projects/project-card";
import { ProjectDetail } from "#/components/features/projects/project-detail";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import {
  useAssignWorktree,
  useCreateProject,
  useCreateWorktree,
  useProject,
  useProjects,
  useRemoveWorktree,
} from "#/hooks/query/use-projects";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import {
  extensionModuleCardGridClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export default function ProjectsPage() {
  const { t } = useTranslation("openhands");
  const { currentPath, navigate } = useNavigation();
  const projectId = projectIdFromPath(currentPath);
  const [createOpen, setCreateOpen] = React.useState(false);
  const listQuery = useProjects();
  const detailQuery = useProject(projectId);
  const createProject = useCreateProject();
  const createWorktree = useCreateWorktree(projectId ?? "");
  const removeWorktree = useRemoveWorktree(projectId ?? "");
  const assignWorktree = useAssignWorktree(projectId ?? "");

  if (projectId) {
    const project = detailQuery.data;
    return (
      <main
        data-testid="projects-page"
        className={kanbanPageScrollShellClassName}
      >
        {project ? (
          <ProjectDetail
            project={project}
            onAddWorktree={(branchName) =>
              createWorktree.mutate({ branch_name: branchName })
            }
            onRemoveWorktree={(worktreeId) => removeWorktree.mutate(worktreeId)}
            onAssignWorktree={(worktreeId, agentSessionId) =>
              assignWorktree.mutate({ worktreeId, agentSessionId })
            }
          />
        ) : null}
      </main>
    );
  }

  const projects = listQuery.data ?? [];

  return (
    <main
      data-testid="projects-page"
      className={kanbanPageScrollShellClassName}
    >
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Typography.H2>{t(I18nKey.PROJECTS$TITLE)}</Typography.H2>
        <BrandButton
          type="button"
          variant="primary"
          testId="projects-create"
          onClick={() => setCreateOpen(true)}
        >
          {t(I18nKey.PROJECTS$CREATE)}
        </BrandButton>
      </header>

      {projects.length === 0 ? (
        <div
          data-testid="projects-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm font-medium text-white">
            {t(I18nKey.PROJECTS$EMPTY)}
          </p>
          <p className="mt-2 text-sm text-tertiary-light">
            {t(I18nKey.PROJECTS$EMPTY_HINT)}
          </p>
        </div>
      ) : (
        <div className={extensionModuleCardGridClassName}>
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onSelect={(selected) => navigate(projectDetailPath(selected.id))}
            />
          ))}
        </div>
      )}

      <CreateProjectModal
        isOpen={createOpen}
        isPending={createProject.isPending}
        onClose={() => setCreateOpen(false)}
        onSubmit={(payload) => {
          createProject.mutate(payload, {
            onSuccess: (created) => {
              setCreateOpen(false);
              navigate(projectDetailPath(created.id));
            },
            onError: () => displayErrorToast(t(I18nKey.ERROR$GENERIC)),
          });
        }}
      />
    </main>
  );
}

export { PROJECTS_PATH };
