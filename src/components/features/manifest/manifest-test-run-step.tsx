import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";

interface SetupTestRunStepProps {
  run: AutomationRun | null;
  error: string | null;
}

function isFailedStatus(status: AutomationRunStatus): boolean {
  return (
    status === AutomationRunStatus.FAILED ||
    status === AutomationRunStatus.CANCELLED ||
    status === AutomationRunStatus.SKIPPED
  );
}

/** The controlled first run that keeps a new automation disabled until proven. */
export function SetupTestRunStep({ run, error }: SetupTestRunStepProps) {
  const { t } = useTranslation("openhands");
  const failed = run ? isFailedStatus(run.status) : false;
  const failureDetail = run?.error_detail?.trim() || error;

  return (
    <div className="flex flex-col gap-4" data-testid="setup-test-run">
      <p className="text-sm text-[var(--oh-muted)]">
        {t(I18nKey.SETUP$TEST_DESCRIPTION)}
      </p>

      {!run && !error && (
        <p className="text-sm">{t(I18nKey.SETUP$TEST_READY)}</p>
      )}

      {run && (
        <div className="flex flex-col gap-2 rounded-lg border border-[var(--oh-border)] p-4">
          <RunStatusBadge status={run.status} />
          {run.phase_label && (
            <p className="text-sm text-[var(--oh-muted)]">{run.phase_label}</p>
          )}

          {run.status === AutomationRunStatus.COMPLETED && (
            <p className="text-sm" data-testid="setup-test-success">
              {t(I18nKey.SETUP$TEST_SUCCESS)}
            </p>
          )}

          {run.conversation_id && (
            <NavigationLink
              to={`/conversations/${run.conversation_id}`}
              className="text-sm underline"
            >
              {t(I18nKey.FEATURED_AUTOMATIONS$VIEW_CONVERSATION)}
            </NavigationLink>
          )}
        </div>
      )}

      {(failed || error) && (
        <div className="flex flex-col gap-2" role="alert">
          <p className="text-sm font-medium text-danger">
            {t(I18nKey.SETUP$TEST_FAILED)}
          </p>
          {failureDetail && (
            <pre
              className="whitespace-pre-wrap break-words rounded-lg bg-surface-raised p-3 text-xs"
              data-testid="setup-test-error"
            >
              {failureDetail}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
