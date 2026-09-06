import React from "react";
import { useTranslation } from "react-i18next";
import {
  LOOPS_PATH,
  loopRunIdFromPath,
} from "#/api/loop-service/loop-constants";
import { LoopRunTimeline } from "#/components/features/loops/loop-run-timeline";
import { LoopsOverview } from "#/components/features/loops/loops-overview";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { useLoopRun } from "#/hooks/query/use-loops";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";

export default function LoopsPage() {
  const { t } = useTranslation("openhands");
  const { currentPath, navigate } = useNavigation();
  const runId = loopRunIdFromPath(currentPath);
  const runQuery = useLoopRun(runId);

  if (runId) {
    return (
      <main data-testid="loops-page" className={kanbanPageScrollShellClassName}>
        <BrandButton
          type="button"
          variant="tertiary"
          testId="loops-back"
          onClick={() => navigate(LOOPS_PATH)}
        >
          {t(I18nKey.LOOPS$BACK)}
        </BrandButton>
        {runQuery.data ? (
          <div className="mt-4">
            <LoopRunTimeline run={runQuery.data} />
          </div>
        ) : null}
      </main>
    );
  }

  return (
    <main data-testid="loops-page" className={kanbanPageScrollShellClassName}>
      <header className="mb-4">
        <Typography.H2>{t(I18nKey.LOOPS$TITLE)}</Typography.H2>
      </header>
      <LoopsOverview />
    </main>
  );
}

export { LOOPS_PATH };
