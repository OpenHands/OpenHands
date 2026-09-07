import { useTranslation } from "react-i18next";
import { PROJECT_STATUSES } from "#/api/projects-service/projects-constants";
import type {
  ProjectStatus,
  ProjectSummary,
} from "#/api/projects-service/projects-types";
import { I18nKey } from "#/i18n/declaration";
import { formatUsd } from "#/components/features/kanban/kanban-cost";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { formControlTransitionClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";

const STATUS_KEY: Record<ProjectStatus, I18nKey> = {
  active: I18nKey.PROJECTS$STATUS_ACTIVE,
  idle: I18nKey.PROJECTS$STATUS_IDLE,
  error: I18nKey.PROJECTS$STATUS_ERROR,
};

const STATUS_CLASS: Record<ProjectStatus, string> = {
  active: "text-green-400",
  idle: "text-tertiary-light",
  error: "text-red-400",
};

export interface ProjectCardProps {
  project: ProjectSummary;
  onSelect?: (project: ProjectSummary) => void;
}

export function ProjectCard({ project, onSelect }: ProjectCardProps) {
  const { t } = useTranslation("openhands");
  const status = PROJECT_STATUSES.includes(project.status)
    ? project.status
    : "idle";

  return (
    <button
      type="button"
      data-testid={`project-card-${project.id}`}
      onClick={() => onSelect?.(project)}
      className={cn(
        "w-full rounded-xl bg-base-secondary p-4 text-left",
        formControlTransitionClassName,
        "hover:bg-[var(--oh-interactive-hover)] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/20",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="min-w-0 break-words text-sm font-medium leading-5 text-white">
          {project.name}
        </span>
        <span
          data-testid={`project-card-status-${project.id}`}
          className={cn(extensionModuleCardPillClassName, STATUS_CLASS[status])}
        >
          {t(STATUS_KEY[status])}
        </span>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs leading-4 text-tertiary-light">
        <span data-testid={`project-card-branches-${project.id}`}>
          {t(I18nKey.PROJECTS$BRANCHES)}
          <span className="ml-1 tabular-nums text-white">
            {project.worktree_count}
          </span>
        </span>
        <span
          data-testid={`project-card-cost-${project.id}`}
          className="tabular-nums text-white"
        >
          {t(I18nKey.PROJECTS$COST)}
          <span className="ml-1">{formatUsd(project.cost_cap)}</span>
        </span>
      </div>
    </button>
  );
}
