import React from "react";
import { useTranslation } from "react-i18next";
import type { RoutingTrace } from "#/api/routing-service/routing-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

export interface DecisionTraceDrawerProps {
  trace: RoutingTrace | null;
  onClose: () => void;
}

export function DecisionTraceDrawer({
  trace,
  onClose,
}: DecisionTraceDrawerProps) {
  const { t } = useTranslation("openhands");
  if (!trace) return null;
  return (
    <aside
      data-testid="routing-decision-trace"
      className="rounded-lg border border-[var(--oh-border)] bg-base-secondary p-4"
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">
          {t(I18nKey.ROUTING$TRACE)}
        </h3>
        <BrandButton
          type="button"
          variant="tertiary"
          testId="routing-trace-close"
          onClick={onClose}
        >
          {t(I18nKey.ROUTING$CLOSE)}
        </BrandButton>
      </div>
      <p data-testid="routing-trace-task" className="text-sm text-white">
        {trace.task_text}
      </p>
      <p
        data-testid="routing-trace-classification"
        className="mt-2 text-xs text-[var(--oh-muted)]"
      >
        {t(I18nKey.ROUTING$CLASSIFICATION)}: {trace.classification.work_type}/
        {trace.classification.sensitivity} ({trace.classifier_version})
      </p>
      <ul
        data-testid="routing-trace-filters"
        className="mt-2 text-xs text-[var(--oh-muted)]"
      >
        {trace.filters.map((item) => (
          <li key={`${item.id}-${item.reason}`}>
            {item.id}: {item.reason}
          </li>
        ))}
      </ul>
      <ol
        data-testid="routing-trace-ranked"
        className="mt-2 text-xs text-white"
      >
        {trace.ranked.map((item) => (
          <li key={item.id}>
            {item.provider_key}/{item.id} {item.score} {item.score_source}
          </li>
        ))}
      </ol>
      <p data-testid="routing-trace-chosen" className="mt-2 text-sm text-white">
        {t(I18nKey.ROUTING$CHOSEN)}: {trace.chosen.provider_key}/
        {trace.chosen.model}
      </p>
      <p
        data-testid="routing-trace-reason"
        className="mt-2 whitespace-pre-wrap text-sm text-white"
      >
        {trace.reason}
      </p>
    </aside>
  );
}
