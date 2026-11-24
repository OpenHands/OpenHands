import { FolderKanban } from "lucide-react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { useProjectNames } from "#/hooks/query/use-project-names";
import { projectIdKey } from "#/api/projects-service/projects-service.api";
import { projectPath } from "#/utils/project-paths";

/**
 * The conversation's project, from its `project` tag, as a link to the
 * project page. Hidden while Projects is turned off.
 */
export function ProjectBadge({
  tags,
}: {
  tags?: Record<string, string> | null;
}) {
  const { t } = useTranslation("openhands");
  const { names, enabled } = useProjectNames();
  const projectId = tags?.project;
  if (!enabled || !projectId) return null;
  const name = names.get(projectIdKey(projectId)) ?? projectId.slice(0, 8);
  return (
    <NavigationLink
      to={projectPath(projectId)}
      data-testid="conversation-project-badge"
      title={t(I18nKey.PROJECTS$OPEN_PROJECT, { name })}
      className="inline-flex max-w-48 shrink-0 items-center gap-1.5 rounded-md border border-control bg-base-secondary px-1.5 py-0.5 text-2xs font-medium leading-4 text-muted hover:text-foreground"
    >
      <FolderKanban size={13} aria-hidden="true" />
      <span className="truncate">{name}</span>
    </NavigationLink>
  );
}
