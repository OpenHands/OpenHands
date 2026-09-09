import React from "react";
import { useTranslation } from "react-i18next";
import type { GraphIndexStatus } from "#/api/graph-service/graph-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export interface IndexerStatusCardProps {
  status: GraphIndexStatus;
  onClear: () => void;
  onRetrigger: () => void;
  isBusy?: boolean;
}

export function IndexerStatusCard({
  status,
  onClear,
  onRetrigger,
  isBusy = false,
}: IndexerStatusCardProps) {
  const { t } = useTranslation("openhands");
  const [confirmClear, setConfirmClear] = React.useState(false);
  const [confirmRetrigger, setConfirmRetrigger] = React.useState(false);
  const coverage = Math.round((status.coverage || 0) * 100);

  return (
    <section
      data-testid="graph-indexer-status"
      className="rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.GRAPH$STATUS)}</Typography>
      <p
        data-testid="graph-indexer-running"
        className="mt-2 text-sm text-white"
      >
        {status.running ? t(I18nKey.GRAPH$RUNNING) : t(I18nKey.GRAPH$IDLE)}
      </p>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div>
          <dt className="text-tertiary-light">{t(I18nKey.GRAPH$FILES)}</dt>
          <dd data-testid="graph-files-indexed">{status.files_indexed}</dd>
        </div>
        <div>
          <dt className="text-tertiary-light">{t(I18nKey.GRAPH$SYMBOLS)}</dt>
          <dd data-testid="graph-symbols">{status.symbols}</dd>
        </div>
        <div>
          <dt className="text-tertiary-light">{t(I18nKey.GRAPH$EDGES)}</dt>
          <dd data-testid="graph-edges">{status.edges}</dd>
        </div>
        <div>
          <dt className="text-tertiary-light">{t(I18nKey.GRAPH$COVERAGE)}</dt>
          <dd data-testid="graph-coverage">{`${coverage}%`}</dd>
        </div>
      </dl>
      <p
        data-testid="graph-languages"
        className="mt-2 text-xs text-tertiary-light"
      >
        {`${t(I18nKey.GRAPH$LANGUAGES)}: ${status.languages_used.join(", ")}`}
      </p>
      <p
        data-testid="graph-last-index"
        className="mt-1 text-xs text-tertiary-light"
      >
        {`${t(I18nKey.GRAPH$LAST_INDEX)}: ${status.last_full_index_at || t(I18nKey.GRAPH$NONE)}`}
      </p>
      {status.last_error ? (
        <p data-testid="graph-last-error" className="mt-1 text-xs text-red-400">
          {`${t(I18nKey.GRAPH$LAST_ERROR)}: ${status.last_error}`}
        </p>
      ) : null}
      <div className="mt-4 flex gap-2">
        <BrandButton
          type="button"
          variant="secondary"
          testId="graph-retrigger"
          isDisabled={isBusy}
          onClick={() => setConfirmRetrigger(true)}
        >
          {t(I18nKey.GRAPH$REINDEX)}
        </BrandButton>
        <BrandButton
          type="button"
          variant="danger"
          testId="graph-clear"
          isDisabled={isBusy}
          onClick={() => setConfirmClear(true)}
        >
          {t(I18nKey.GRAPH$CLEAR)}
        </BrandButton>
      </div>
      {confirmClear ? (
        <ConfirmationModal
          text={t(I18nKey.GRAPH$CLEAR_CONFIRM)}
          onConfirm={() => {
            setConfirmClear(false);
            onClear();
          }}
          onCancel={() => setConfirmClear(false)}
        />
      ) : null}
      {confirmRetrigger ? (
        <ConfirmationModal
          text={t(I18nKey.GRAPH$REINDEX_CONFIRM)}
          onConfirm={() => {
            setConfirmRetrigger(false);
            onRetrigger();
          }}
          onCancel={() => setConfirmRetrigger(false)}
        />
      ) : null}
    </section>
  );
}
