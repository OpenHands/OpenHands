import React from "react";
import { useTranslation } from "react-i18next";
import {
  DEFAULT_PROJECT_BOARD_NAME,
  KANBAN_PATH,
} from "#/api/kanban-service/kanban-constants";
import type { SuggestedKanbanCard } from "#/api/kanban-service/kanban-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { useInitProject, usePreviewProject } from "#/hooks/query/use-kanban";
import { useKanbanWorkspace } from "#/hooks/use-kanban-workspace";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";
import { cn } from "#/utils/utils";

export function ProjectInitForm() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const { selected } = useKanbanWorkspace();
  const [spec, setSpec] = React.useState("");
  const [suggested, setSuggested] = React.useState<SuggestedKanbanCard[]>([]);
  const preview = usePreviewProject();
  const init = useInitProject();

  return (
    <main
      data-testid="project-init-page"
      className={kanbanPageScrollShellClassName}
    >
      <header className="mb-6 shrink-0">
        <Typography.H2>{t(I18nKey.PROJECT_INIT$TITLE)}</Typography.H2>
      </header>
      <form
        className="flex max-w-2xl flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault();
        }}
      >
        <label className="flex flex-col gap-2 text-sm text-tertiary-light">
          {t(I18nKey.PROJECT_INIT$SPEC_LABEL)}
          <textarea
            data-testid="project-init-spec"
            value={spec}
            onChange={(event) => setSpec(event.target.value)}
            placeholder={t(I18nKey.PROJECT_INIT$SPEC_PLACEHOLDER)}
            className={cn(formControlMultilineFieldClassName, "min-h-40")}
          />
        </label>
        <div className="flex flex-wrap gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            testId="project-init-scan"
            isDisabled={preview.isPending}
            onClick={() => {
              preview.mutate(
                { spec, root: selected?.path },
                { onSuccess: (data) => setSuggested(data.suggested) },
              );
            }}
          >
            {t(I18nKey.PROJECT_INIT$SCAN)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            testId="project-init-create"
            isDisabled={init.isPending || suggested.length === 0}
            onClick={() => {
              init.mutate(
                {
                  spec,
                  board_name: DEFAULT_PROJECT_BOARD_NAME,
                  root: selected?.path,
                },
                { onSuccess: () => navigate(KANBAN_PATH) },
              );
            }}
          >
            {t(I18nKey.PROJECT_INIT$CREATE)}
          </BrandButton>
        </div>
      </form>

      <section className="mt-8 max-w-2xl" data-testid="project-init-preview">
        <h2 className="mb-3 text-sm font-medium text-white">
          {t(I18nKey.PROJECT_INIT$PREVIEW)}
        </h2>
        {suggested.length === 0 ? (
          <p
            data-testid="project-init-empty"
            className="text-sm text-tertiary-light"
          >
            {t(I18nKey.PROJECT_INIT$NO_CARDS)}
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {suggested.map((card) => (
              <li
                key={`${card.source}-${card.title}`}
                data-testid="project-init-suggested-card"
                className="rounded-xl bg-base-secondary px-3 py-2.5"
              >
                <p className="text-sm font-medium text-white">{card.title}</p>
                <p className="mt-1 text-xs text-tertiary-light">
                  {t(I18nKey.PROJECT_INIT$SOURCE, { source: card.source })}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
