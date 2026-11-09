import React from "react";
import { useSearchParams } from "react-router";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { NewProjectModal } from "#/components/features/projects/new-project-modal";
import { ProjectsOffNotice } from "#/components/features/projects/projects-off-notice";
import { useProjects, useProjectsEnabled } from "#/hooks/query/use-projects";
import { useNavigation } from "#/context/navigation-context";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import { projectPath } from "#/utils/project-paths";

/** /projects: the active backend's projects, and New project. */
export default function ProjectsPage() {
  const { t } = useTranslation("openhands");
  const enabled = useProjectsEnabled();
  const { data: projects, isLoading, isError } = useProjects();
  const { navigate } = useNavigation();
  const [searchParams, setSearchParams] = useSearchParams();
  // The sidebar's "+" opens this page with ?new=1.
  const isCreating = enabled && searchParams.get("new") === "1";
  const setIsCreating = (open: boolean) =>
    setSearchParams(
      (params) => {
        const next = new URLSearchParams(params);
        if (open) next.set("new", "1");
        else next.delete("new");
        return next;
      },
      { replace: true },
    );

  let body: React.ReactNode;
  if (!enabled) {
    body = <ProjectsOffNotice />;
  } else if (isLoading) {
    body = <p className="text-sm text-muted">{t(I18nKey.PROJECTS$LOADING)}</p>;
  } else if (isError) {
    body = (
      <p className="text-sm text-danger">{t(I18nKey.PROJECTS$LOAD_FAILED)}</p>
    );
  } else if (!projects?.length) {
    body = (
      <p data-testid="projects-empty" className="text-sm text-muted">
        {t(I18nKey.PROJECTS$EMPTY)}
      </p>
    );
  } else {
    body = (
      <ul className="flex flex-col gap-2" data-testid="projects-list">
        {[...projects].reverse().map((project) => (
          <li key={project.id}>
            <NavigationLink
              to={projectPath(project.id)}
              data-testid="project-row"
              className="flex flex-col gap-1 rounded-lg border border-control bg-base-secondary px-4 py-3 hover:bg-contrast/5"
            >
              <span className="text-sm font-medium text-foreground">
                {project.name}
              </span>
              {project.description ? (
                <span className="line-clamp-2 text-sm text-muted">
                  {project.description}
                </span>
              ) : null}
            </NavigationLink>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <div data-testid="projects-page" className="flex h-full">
      <main className={settingsLikeMainScrollClassName}>
        <div className="mx-auto flex w-full min-w-0 max-w-200 flex-col gap-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 flex-1 basis-64 space-y-1">
              <h1 className="text-xl font-medium leading-6 text-foreground">
                {t(I18nKey.PROJECTS$TITLE)}
              </h1>
              <p className="max-w-2xl text-sm text-muted">
                {t(I18nKey.PROJECTS$DESCRIPTION)}
              </p>
            </div>
            {enabled ? (
              <BrandButton
                testId="new-project-button"
                type="button"
                variant="primary"
                startContent={<Plus width={16} height={16} />}
                onClick={() => setIsCreating(true)}
              >
                {t(I18nKey.PROJECTS$NEW)}
              </BrandButton>
            ) : null}
          </div>
          {body}
        </div>
      </main>
      {isCreating ? (
        <NewProjectModal
          onClose={() => setIsCreating(false)}
          onCreated={(project) => {
            setIsCreating(false);
            navigate(projectPath(project.id));
          }}
        />
      ) : null}
    </div>
  );
}
