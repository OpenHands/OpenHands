import { useTranslation } from "react-i18next";
import { RunPhase } from "#/components/features/automations/detail/run-phase";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { I18nKey } from "#/i18n/declaration";
import type { AutomationRun } from "#/types/automation";

interface SetupTestRunStepProps {
  run: AutomationRun | null;
  description: string;
}

/**
 * The setup-time view of one controlled automation run. Execution, polling and
 * status semantics remain owned by the existing automation service/hooks; this
 * component only keeps the setup flow from inventing a second representation.
 */
export function SetupTestRunStep({ run, description }: SetupTestRunStepProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="setup-test-run" className="flex flex-col gap-4">
      <p className="text-sm text-[var(--oh-muted)]">{description}</p>

      {run ? (
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

          {run.error_detail ? (
            <p
              role="alert"
              data-testid="setup-test-run-error"
              className="whitespace-pre-wrap break-words text-sm text-red-400"
            >
              {run.error_detail}
            </p>
          ) : null}

          {run.conversation_id ? (
            <a
              data-testid="setup-test-run-conversation"
              href={`/conversations/${run.conversation_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="w-fit text-sm text-muted underline transition-colors hover:text-foreground"
            >
              {t(I18nKey.AUTOMATIONS$IMPORT_VIEW)}
            </a>
          ) : null}
        </div>
      ) : (
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.AUTOMATIONS$RUN_NOW)}
        </p>
      )}
    </div>
  );
}
