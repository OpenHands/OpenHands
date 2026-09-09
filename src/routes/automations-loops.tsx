import { useTranslation } from "react-i18next";
import {
  LOOPS_PATH,
  loopRunIdFromPath,
} from "#/api/loop-service/loop-constants";
import { useAutomationSubPageNav } from "#/components/features/automations/dashboard/use-automation-sub-page-nav";
import { LoopRunTimeline } from "#/components/features/loops/loop-run-timeline";
import { LoopsOverview } from "#/components/features/loops/loops-overview";
import { ManifestSubpageLayout } from "#/components/features/manifest/manifest-subpage-layout";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { useLoopRun } from "#/hooks/query/use-loops";
import { I18nKey } from "#/i18n/declaration";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";

export default function AutomationsLoopsPage() {
  const { t } = useTranslation("openhands");
  const nav = useAutomationSubPageNav();
  const { currentPath, navigate } = useNavigation();
  const runId = loopRunIdFromPath(currentPath);
  const runQuery = useLoopRun(runId);

  const body = runId ? (
    <div data-testid="loops-page" className="flex flex-col gap-4">
      <BrandButton
        type="button"
        variant="tertiary"
        testId="loops-back"
        onClick={() => navigate(LOOPS_PATH)}
      >
        {t(I18nKey.LOOPS$BACK)}
      </BrandButton>
      {runQuery.data ? <LoopRunTimeline run={runQuery.data} /> : null}
    </div>
  ) : (
    <div data-testid="loops-page" className="flex flex-col gap-6">
      <h1 className="text-xl font-semibold text-content">
        {t(I18nKey.LOOPS$TITLE)}
      </h1>
      <LoopsOverview />
    </div>
  );

  if (nav) {
    return (
      <ManifestSubpageLayout
        heading={nav.heading}
        navTestIdBase="automations-navbar"
        items={nav.items}
      >
        {body}
      </ManifestSubpageLayout>
    );
  }

  return (
    <main className={settingsLikeMainScrollClassName}>
      <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
        {body}
      </div>
    </main>
  );
}
