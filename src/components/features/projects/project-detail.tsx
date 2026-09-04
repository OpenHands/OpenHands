import { useTranslation } from "react-i18next";
import { KANBAN_PATH } from "#/api/kanban-service/kanban-constants";
import { PROJECTS_PATH } from "#/api/projects-service/projects-constants";
import type { Project } from "#/api/projects-service/projects-types";
import { formatUsd } from "#/components/features/kanban/kanban-cost";
import { WorktreePanel } from "#/components/features/projects/worktree-panel";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

const PROJECT_STATUS_KEY: Record<string, I18nKey> = {
  active: I18nKey.PROJECTS$STATUS_ACTIVE,
  idle: I18nKey.PROJECTS$STATUS_IDLE,
  error: I18nKey.PROJECTS$STATUS_ERROR,
};

export interface ProjectDetailProps {
  project: Project;
  onAddWorktree?: (branchName: string) => void;
  onRemoveWorktree?: (worktreeId: string) => void;
  onAssignWorktree?: (worktreeId: string, agentSessionId: string) => void;
}

export function ProjectDetail({
  project,
  onAddWorktree,
  onRemoveWorktree,
  onAssignWorktree,
}: ProjectDetailProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();

  return (
    <div data-testid="project-detail" className="flex flex-col gap-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <Typography.H2>{project.name}</Typography.H2>
        <BrandButton
          type="button"
          variant="secondary"
          testId="project-detail-back"
          onClick={() => navigate(PROJECTS_PATH)}
        >
          {t(I18nKey.PROJECTS$BACK)}
        </BrandButton>
      </header>

      <section data-testid="project-metadata">
        <h2 className="mb-2 text-sm font-medium text-white">
          {t(I18nKey.PROJECTS$METADATA)}
        </h2>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-sm">
          <dt className="text-tertiary-light">{t(I18nKey.PROJECTS$STATUS)}</dt>
          <dd className="text-white">
            {t(
              PROJECT_STATUS_KEY[project.status] ??
                I18nKey.PROJECTS$STATUS_IDLE,
            )}
          </dd>
          <dt className="text-tertiary-light">
            {t(I18nKey.PROJECTS$DEFAULT_BRANCH)}
          </dt>
          <dd className="text-white">{project.default_branch}</dd>
          <dt className="text-tertiary-light">
            {t(I18nKey.PROJECTS$LOCAL_PATH)}
          </dt>
          <dd className="break-all text-white">{project.local_path}</dd>
          <dt className="text-tertiary-light">
            {t(I18nKey.PROJECTS$COST_CAP)}
          </dt>
          <dd className="tabular-nums text-white">
            {formatUsd(project.cost_cap)}
          </dd>
          {project.description ? (
            <>
              <dt className="text-tertiary-light">
                {t(I18nKey.PROJECTS$DESCRIPTION)}
              </dt>
              <dd className="text-white">{project.description}</dd>
            </>
          ) : null}
          <dt className="text-tertiary-light">
            {t(I18nKey.PROJECTS$KANBAN_BOARD)}
          </dt>
          <dd>
            {project.kanban_board_id ? (
              <BrandButton
                type="button"
                variant="secondary"
                testId="project-open-kanban"
                onClick={() => navigate(KANBAN_PATH)}
              >
                {t(I18nKey.PROJECTS$OPEN_KANBAN)}
              </BrandButton>
            ) : (
              <span
                data-testid="project-no-kanban"
                className="text-tertiary-light"
              >
                {t(I18nKey.PROJECTS$NO_KANBAN)}
              </span>
            )}
          </dd>
        </dl>
      </section>

      <WorktreePanel
        worktrees={project.worktrees}
        onAdd={onAddWorktree}
        onRemove={onRemoveWorktree}
        onAssign={onAssignWorktree}
      />
    </div>
  );
}
