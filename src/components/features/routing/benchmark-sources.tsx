import React from "react";
import { useTranslation } from "react-i18next";
import type { RoutingSourcesResponse } from "#/api/routing-service/routing-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

export interface BenchmarkSourcesProps {
  sources: RoutingSourcesResponse;
  ingesting?: boolean;
  onIngest: (sourceId?: string) => void;
}

export function BenchmarkSources({
  sources,
  ingesting,
  onIngest,
}: BenchmarkSourcesProps) {
  const { t } = useTranslation("openhands");
  return (
    <section
      data-testid="routing-benchmark-sources"
      className="flex flex-col gap-3"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">
          {t(I18nKey.ROUTING$SOURCES)}
        </h3>
        <BrandButton
          type="button"
          variant="secondary"
          testId="routing-ingest-now"
          isDisabled={ingesting}
          onClick={() => onIngest()}
        >
          {t(I18nKey.ROUTING$INGEST_NOW)}
        </BrandButton>
      </div>
      <p className="text-xs text-[var(--oh-muted)]">
        {t(I18nKey.ROUTING$PRIVACY_CURATED)}
      </p>
      <ul className="flex flex-col gap-2">
        {sources.sources.map((source) => (
          <li
            key={source.id}
            data-testid={`routing-source-${source.id}`}
            className="rounded-lg border border-[var(--oh-border)] p-3 text-sm text-white"
          >
            <div>{source.id}</div>
            <div className="text-xs text-[var(--oh-muted)]">
              {t(I18nKey.ROUTING$LAST_SUCCESS)}: {source.last_success ?? ""}
            </div>
            {source.last_error ? (
              <div
                data-testid={`routing-source-error-${source.id}`}
                className="text-xs text-red-400"
              >
                {t(I18nKey.ROUTING$LAST_ERROR)}: {source.last_error}
              </div>
            ) : null}
            {source.stale ? (
              <div
                data-testid={`routing-source-stale-${source.id}`}
                className="text-xs text-amber-400"
              >
                {t(I18nKey.ROUTING$STALE)}
              </div>
            ) : null}
            <BrandButton
              type="button"
              variant="tertiary"
              testId={`routing-ingest-${source.id}`}
              onClick={() => onIngest(source.id)}
            >
              {t(I18nKey.ROUTING$INGEST_NOW)}
            </BrandButton>
          </li>
        ))}
      </ul>
    </section>
  );
}
