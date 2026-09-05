import { useTranslation } from "react-i18next";
import { RunPhase } from "#/components/features/automations/detail/run-phase";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";

interface SetupTestRunStepProps {
  run: AutomationRun | null;
  description: string;
}

function isFailedStatus(status: AutomationRunStatus): boolean {
  return (
    status === AutomationRunStatus.FAILED ||
    status === AutomationRunStatus.CANCELLED ||
    status === AutomationRunStatus.SKIPPED
  );
}

/**
 * The setup-time view of one controlled automation run. Execution, polling and
 * status semantics remain owned by the existing automation service/hooks; this
 * component only keeps the setup flow from inventing a second representation.
 */
export function SetupTestRunStep({ run }: SetupTestRunStepProps) {
  const { t } = useTranslation("openhands");
  const failed = run ? isFailedStatus(run.status) : false;

  return (
    <div data-testid="setup-test-run" className="flex flex-col gap-4">
      <p className="text-sm text-[var(--oh-muted)]">
        {t(I18nKey.SETUP$TEST_DESCRIPTION)}
      </p>

      {!run && (
        <p className="text-sm" data-testid="setup-test-run-ready">
          {t(I18nKey.SETUP$TEST_READY)}
        </p>
      )}

      {run && (
        <div className="flex flex-col gap-3 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-4">
          <div className="flex flex-wrap items-center gap-3">
            <RunStatusBadge status={run.status} />
            <RunPhase
              status={run.status}
              code={run.phase_code}
              label={run.phase_label}
              updatedAt={run.phase_updated_at}
              wide
            />
          </div>

          {run.status === AutomationRunStatus.COMPLETED && (
            <p className="text-sm" data-testid="setup-test-run-success">
              {t(I18nKey.SETUP$TEST_SUCCESS)}
            </p>
          )}

          {failed && (
            <div className="flex flex-col gap-2" role="alert">
              <p className="text-sm font-medium text-danger">
                {t(I18nKey.SETUP$TEST_FAILED)}
              </p>
              {run.error_detail ? (
                <pre
                  data-testid="setup-test-run-error"
                  className="whitespace-pre-wrap break-words rounded-lg bg-surface-raised p-3 text-xs"
                >
                  {run.error_detail}
                </pre>
              ) : null}
            </div>
          )}

          {run.conversation_id ? (
            <a
              data-testid="setup-test-run-conversation"
              href={`/conversations/${run.conversation_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="w-fit text-sm text-muted underline transition-colors hover:text-foreground"
            >
              {t(I18nKey.FEATURED_AUTOMATIONS$VIEW_CONVERSATION)}
            </a>
          ) : null}
        </div>
      )}
    </div>
  );
}
