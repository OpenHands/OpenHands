import React from "react";
import { useTranslation } from "react-i18next";
import type { RoutingRegistrySnapshot } from "#/api/routing-service/routing-types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

export interface ModelRegistryBrowserProps {
  registry: RoutingRegistrySnapshot;
}

export function ModelRegistryBrowser({ registry }: ModelRegistryBrowserProps) {
  const { t } = useTranslation("openhands");
  return (
    <section
      data-testid="routing-model-registry"
      className="flex flex-col gap-3"
    >
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$REGISTRY)}
      </h3>
      {registry.privacy_stale ? (
        <p
          data-testid="routing-privacy-stale"
          className="text-sm text-amber-400"
        >
          {t(I18nKey.ROUTING$PRIVACY_STALE)}
        </p>
      ) : null}
      <p className="text-xs text-[var(--oh-muted)]">
        {t(I18nKey.ROUTING$PRIVACY_CURATED)}
      </p>
      <ul className="flex flex-col gap-2">
        {registry.models.map((item) => {
          const coding = item.benchmarks.coding;
          const dimmed = item.reachable === false;
          return (
            <li
              key={item.id}
              data-testid={`routing-model-${item.id}`}
              className={cn(
                "rounded-lg border border-[var(--oh-border)] p-3",
                dimmed && "opacity-50",
              )}
            >
              <div className="flex flex-wrap items-center gap-2 text-sm text-white">
                <span>{item.id}</span>
                <span data-testid={`routing-model-verified-${item.id}`}>
                  {item.verified
                    ? t(I18nKey.ROUTING$VERIFIED)
                    : t(I18nKey.ROUTING$UNVERIFIED)}
                </span>
                <span data-testid={`routing-model-reachable-${item.id}`}>
                  {item.reachable === false
                    ? t(I18nKey.ROUTING$UNREACHABLE)
                    : t(I18nKey.ROUTING$REACHABLE)}
                </span>
              </div>
              <div className="mt-2 text-xs text-[var(--oh-muted)]">
                {t(I18nKey.ROUTING$TRADEOFF_COST)}: {item.cost_per_1k} ·{" "}
                {t(I18nKey.ROUTING$TRADEOFF_LATENCY)}: {item.latency_s_p90}
              </div>
              <div
                data-testid={`routing-privacy-${item.id}`}
                className="mt-1 text-xs text-[var(--oh-muted)]"
              >
                {t(I18nKey.ROUTING$RETENTION)}: {item.retention} ·{" "}
                {t(I18nKey.ROUTING$WATERMARK)}: {item.watermark}
              </div>
              {coding ? (
                <div className="mt-2">
                  <div
                    data-testid={`routing-bench-coding-${item.id}`}
                    title={coding.provenance?.source ?? undefined}
                    className="h-2 rounded bg-surface-raised"
                    style={{ width: `${Math.round(coding.score * 100)}%` }}
                  />
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
