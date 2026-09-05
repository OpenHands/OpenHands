import React from "react";
import { useTranslation } from "react-i18next";
import {
  FEATURE_DEV_PATH,
  featureDevRunIdFromPath,
  featureDevRunPath,
} from "#/api/feature-developer-service/feature-developer-constants";
import type { FeatureDevStatus } from "#/api/feature-developer-service/feature-developer-types";
import { SpecInput } from "#/components/features/feature-developer/spec-input";
import { RunTimeline } from "#/components/features/feature-developer/run-timeline";
import { ReportView } from "#/components/features/feature-developer/report-view";
import { BrandButton } from "#/components/features/settings/brand-button";
import { formatUsd } from "#/components/features/kanban/kanban-cost";
import { useNavigation } from "#/context/navigation-context";
import {
  useFeatureDevReport,
  useFeatureDevRun,
  useFeatureDevRuns,
  useStartFeatureDevRun,
} from "#/hooks/query/use-feature-developer";
import { useProjects } from "#/hooks/query/use-projects";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { cn } from "#/utils/utils";

const STATUS_KEY: Record<FeatureDevStatus, I18nKey> = {
  pending: I18nKey.FEATURE_DEV$STATUS_PENDING,
  running: I18nKey.FEATURE_DEV$STATUS_RUNNING,
  passed: I18nKey.FEATURE_DEV$STATUS_PASSED,
  failed: I18nKey.FEATURE_DEV$STATUS_FAILED,
  partial: I18nKey.FEATURE_DEV$STATUS_PARTIAL,
  aborted: I18nKey.FEATURE_DEV$STATUS_ABORTED,
  paused: I18nKey.FEATURE_DEV$STATUS_PAUSED,
};

const STATUS_CLASS: Record<string, string> = {
  passed: "text-green-400",
  failed: "text-red-400",
  partial: "text-amber-400",
  running: "text-blue-400",
  paused: "text-amber-400",
  aborted: "text-red-400",
  pending: "text-tertiary-light",
};

export default function FeatureDeveloperPage() {
  const { t } = useTranslation("openhands");
  const { currentPath, navigate } = useNavigation();
  const runId = featureDevRunIdFromPath(currentPath);
  const projectsQuery = useProjects();
  const listQuery = useFeatureDevRuns();
  const detailQuery = useFeatureDevRun(runId);
  const reportQuery = useFeatureDevReport(runId);
  const startRun = useStartFeatureDevRun();

  if (runId) {
    const run = detailQuery.data;
    return (
      <main
        data-testid="feature-dev-page"
        className={kanbanPageScrollShellClassName}
      >
        <BrandButton
          type="button"
          variant="tertiary"
          testId="feature-dev-back"
          onClick={() => navigate(FEATURE_DEV_PATH)}
        >
          {t(I18nKey.FEATURE_DEV$BACK)}
        </BrandButton>
        {run ? (
          <div className="mt-4 flex flex-col gap-6">
            <RunTimeline run={run} />
            {reportQuery.data?.markdown ? (
              <ReportView markdown={reportQuery.data.markdown} />
            ) : null}
          </div>
        ) : null}
      </main>
    );
  }

  const runs = listQuery.data ?? [];
  const projects = projectsQuery.data ?? [];

  return (
    <main
      data-testid="feature-dev-page"
      className={kanbanPageScrollShellClassName}
    >
      <header className="mb-4">
        <Typography.H2>{t(I18nKey.FEATURE_DEV$TITLE)}</Typography.H2>
      </header>

      <SpecInput
        projects={projects}
        isPending={startRun.isPending}
        onStart={(payload) => {
          startRun.mutate(payload, {
            onSuccess: (created) => navigate(featureDevRunPath(created.id)),
            onError: () => displayErrorToast(t(I18nKey.ERROR$GENERIC)),
          });
        }}
      />

      <section className="mt-8 flex flex-col gap-3">
        {runs.length === 0 ? (
          <div data-testid="feature-dev-empty">
            <p className="text-sm font-medium text-white">
              {t(I18nKey.FEATURE_DEV$EMPTY)}
            </p>
            <p className="mt-2 text-sm text-tertiary-light">
              {t(I18nKey.FEATURE_DEV$EMPTY_HINT)}
            </p>
          </div>
        ) : (
          <ul className="flex flex-col gap-2">
            {runs.map((run) => (
              <li key={run.id}>
                <button
                  type="button"
                  data-testid={`feature-dev-run-${run.id}`}
                  onClick={() => navigate(featureDevRunPath(run.id))}
                  className="flex w-full items-center justify-between gap-3 rounded-xl bg-base-secondary p-3 text-left"
                >
                  <span
                    className={cn(
                      extensionModuleCardPillClassName,
                      STATUS_CLASS[run.status] ?? "text-white",
                    )}
                  >
                    {t(
                      STATUS_KEY[run.status] ??
                        I18nKey.FEATURE_DEV$STATUS_PENDING,
                    )}
                  </span>
                  <span className="tabular-nums text-white">
                    {formatUsd(run.total_actual_usd)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

export { FEATURE_DEV_PATH };
