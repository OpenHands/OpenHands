import React from "react";
import { useTranslation } from "react-i18next";
import { ROUTING_PRESETS } from "#/api/routing-service/routing-constants";
import type {
  RoutingLocalRuntimes,
  RoutingPreset,
  RoutingRouterModelResponse,
} from "#/api/routing-service/routing-types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

function presetKey(preset: RoutingPreset): I18nKey {
  if (preset === "local") return I18nKey.ROUTING$PRESET_LOCAL;
  if (preset === "best-intelligence") return I18nKey.ROUTING$PRESET_BEST;
  if (preset === "cheapest") return I18nKey.ROUTING$PRESET_CHEAPEST;
  if (preset === "most-secure") return I18nKey.ROUTING$PRESET_SECURE;
  return I18nKey.ROUTING$PRESET_CUSTOM;
}

export interface RouterModelChooserProps {
  routerModel: RoutingRouterModelResponse;
  localRuntimes?: RoutingLocalRuntimes;
  onSelectPreset: (preset: RoutingPreset) => void;
}

export function RouterModelChooser({
  routerModel,
  localRuntimes,
  onSelectPreset,
}: RouterModelChooserProps) {
  const { t } = useTranslation("openhands");
  const installed = new Set(
    (localRuntimes?.runtimes.ollama?.models ?? []).map((name) =>
      name.toLowerCase(),
    ),
  );
  const pool =
    routerModel.config.preset === "local"
      ? routerModel.resolved.pool.filter((item) => {
          const short = item.id.split("/").slice(1).join("/").toLowerCase();
          return [...installed].some(
            (name) => short.includes(name) || name.includes(short),
          );
        })
      : routerModel.resolved.pool;

  return (
    <section
      data-testid="routing-router-model-chooser"
      className="flex flex-col gap-3"
    >
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$ROUTER_MODEL)}
      </h3>
      <div className="grid gap-2 md:grid-cols-2">
        {ROUTING_PRESETS.map((preset) => {
          const selected = routerModel.config.preset === preset;
          const tradeoffs =
            selected || preset === routerModel.resolved.preset
              ? routerModel.resolved.tradeoffs
              : routerModel.resolved.tradeoffs;
          return (
            <button
              key={preset}
              type="button"
              data-testid={`routing-preset-${preset}`}
              onClick={() => onSelectPreset(preset)}
              className={cn(
                "rounded-lg border p-3 text-left",
                selected
                  ? "border-white bg-surface-raised"
                  : "border-[var(--oh-border)] bg-base-secondary",
              )}
            >
              <div className="text-sm text-white">{t(presetKey(preset))}</div>
              <dl className="mt-2 grid grid-cols-2 gap-1 text-xs text-[var(--oh-muted)]">
                <dt>{t(I18nKey.ROUTING$TRADEOFF_COST)}</dt>
                <dd data-testid={`routing-preset-${preset}-cost`}>
                  {tradeoffs.cost_per_1k ?? ""}
                </dd>
                <dt>{t(I18nKey.ROUTING$TRADEOFF_LATENCY)}</dt>
                <dd>{tradeoffs.latency_s_p90 ?? ""}</dd>
                <dt>{t(I18nKey.ROUTING$TRADEOFF_PRIVACY)}</dt>
                <dd>
                  {tradeoffs.retention}/{tradeoffs.watermark}
                </dd>
                <dt>{t(I18nKey.ROUTING$TRADEOFF_ACCURACY)}</dt>
                <dd>{tradeoffs.classification_accuracy_proxy}</dd>
                <dt>{t(I18nKey.ROUTING$TRADEOFF_OFFLINE)}</dt>
                <dd>{String(tradeoffs.offline_capable)}</dd>
              </dl>
            </button>
          );
        })}
      </div>
      <ul data-testid="routing-preset-pool" className="flex flex-col gap-1">
        {pool.map((item) => (
          <li
            key={item.id}
            data-testid={`routing-pool-${item.id}`}
            className="rounded border border-[var(--oh-border)] px-2 py-1 text-xs text-white"
          >
            {item.provider_key}/{item.id}
            {item.verified
              ? ` · ${t(I18nKey.ROUTING$VERIFIED)}`
              : ` · ${t(I18nKey.ROUTING$UNVERIFIED)}`}
          </li>
        ))}
      </ul>
    </section>
  );
}
