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
import { I18nKey } from "#/i18n/declaration";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";

export function ProjectInitForm() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [spec, setSpec] = React.useState("");
  const [suggested, setSuggested] = React.useState<SuggestedKanbanCard[]>([]);
  const preview = usePreviewProject();
  const init = useInitProject();

  return (
    <main
      data-testid="project-init-page"
      className={settingsLikeMainScrollClassName}
    >
      <h1 className="text-2xl font-semibold">
        {t(I18nKey.PROJECT_INIT$TITLE)}
      </h1>
      <form
        className="mt-6 flex max-w-2xl flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault();
        }}
      >
        <label className="flex flex-col gap-2 text-sm">
          {t(I18nKey.PROJECT_INIT$SPEC_LABEL)}
          <textarea
            data-testid="project-init-spec"
            value={spec}
            onChange={(event) => setSpec(event.target.value)}
            placeholder={t(I18nKey.PROJECT_INIT$SPEC_PLACEHOLDER)}
            className="min-h-40 rounded-md border border-[var(--oh-border)] bg-transparent px-3 py-2"
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
                { spec },
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
                { spec, board_name: DEFAULT_PROJECT_BOARD_NAME },
                { onSuccess: () => navigate(KANBAN_PATH) },
              );
            }}
          >
            {t(I18nKey.PROJECT_INIT$CREATE)}
          </BrandButton>
        </div>
      </form>

      <section className="mt-8 max-w-2xl" data-testid="project-init-preview">
        <h2 className="mb-3 text-lg font-medium">
          {t(I18nKey.PROJECT_INIT$PREVIEW)}
        </h2>
        {suggested.length === 0 ? (
          <p data-testid="project-init-empty">
            {t(I18nKey.PROJECT_INIT$NO_CARDS)}
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {suggested.map((card) => (
              <li
                key={`${card.source}-${card.title}`}
                data-testid="project-init-suggested-card"
                className="rounded-md border border-[var(--oh-border)] px-3 py-2"
              >
                <p className="font-medium">{card.title}</p>
                <p className="text-sm text-[var(--oh-muted)]">
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
