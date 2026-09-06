import React from "react";
import { useTranslation } from "react-i18next";
import {
  ROUTING_GOALS,
  ROUTING_TARGET_AUTO,
} from "#/api/routing-service/routing-constants";
import type {
  RoutingConfig,
  RoutingGoal,
  RoutingRoute,
} from "#/api/routing-service/routing-types";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

function isAutoTarget(target: RoutingRoute["target"]): boolean {
  return target === ROUTING_TARGET_AUTO;
}

function routeTitleKey(route: RoutingRoute): I18nKey {
  if (!route.work_type && route.sensitivity === "sensitive-ip") {
    return I18nKey.ROUTING$ROUTE_SENSITIVE_IP;
  }
  if (route.work_type === "ux") return I18nKey.ROUTING$ROUTE_UX;
  if (route.work_type === "copy") return I18nKey.ROUTING$ROUTE_COPY;
  if (!route.work_type && !route.sensitivity)
    return I18nKey.ROUTING$ROUTE_DEFAULT;
  return I18nKey.ROUTING$ROUTES;
}

function goalKey(goal: RoutingGoal): I18nKey {
  if (goal === "cost") return I18nKey.ROUTING$GOAL_COST;
  if (goal === "speed") return I18nKey.ROUTING$GOAL_SPEED;
  if (goal === "privacy") return I18nKey.ROUTING$GOAL_PRIVACY;
  return I18nKey.ROUTING$GOAL_QUALITY;
}

export interface RoutesTableProps {
  config: RoutingConfig;
  previews?: Record<string, string>;
  onChange: (patch: Partial<RoutingConfig>) => void;
}

export function RoutesTable({ config, previews, onChange }: RoutesTableProps) {
  const { t } = useTranslation("openhands");
  const goalItems = ROUTING_GOALS.map((goal) => ({
    key: goal,
    label: t(goalKey(goal)),
  }));

  const updateRoute = (routeId: string, patch: Partial<RoutingRoute>) => {
    onChange({
      routes: config.routes.map((route) =>
        route.id === routeId ? { ...route, ...patch } : route,
      ),
    });
  };

  return (
    <section data-testid="routing-routes-table" className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$ROUTES)}
      </h3>
      <div className="flex flex-col gap-2">
        {config.routes.map((route) => {
          const auto = isAutoTarget(route.target);
          const preview = previews?.[route.id];
          return (
            <article
              key={route.id}
              data-testid={`routing-route-${route.id}`}
              className="rounded-lg border border-[var(--oh-border)] bg-base-secondary p-3 flex flex-col gap-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm text-white">
                  {t(routeTitleKey(route))}
                </span>
                {route.work_type ? (
                  <span
                    data-testid={`routing-chip-work-${route.id}`}
                    className="rounded-full border border-[var(--oh-border)] px-2 py-0.5 text-xs text-[var(--oh-muted)]"
                  >
                    {route.work_type}
                  </span>
                ) : null}
                {route.sensitivity ? (
                  <span
                    data-testid={`routing-chip-sensitivity-${route.id}`}
                    className="rounded-full border border-[var(--oh-border)] px-2 py-0.5 text-xs text-[var(--oh-muted)]"
                  >
                    {route.sensitivity}
                  </span>
                ) : null}
                {auto ? (
                  <span
                    data-testid={`routing-target-auto-${route.id}`}
                    className="text-xs text-[var(--oh-muted)]"
                  >
                    {preview
                      ? t(I18nKey.ROUTING$PREVIEW, {
                          provider: preview.split("/")[0],
                          model: preview.split("/").slice(1).join("/"),
                        })
                      : t(I18nKey.ROUTING$AUTO)}
                  </span>
                ) : (
                  <span
                    data-testid={`routing-locked-${route.id}`}
                    className={cn(
                      "rounded-full bg-surface-raised px-2 py-0.5 text-xs text-white",
                    )}
                  >
                    {t(I18nKey.ROUTING$LOCKED)}
                  </span>
                )}
              </div>
              <SettingsDropdownInput
                testId={`routing-goal-${route.id}`}
                name={`routing-goal-${route.id}`}
                label={t(I18nKey.ROUTING$GOAL)}
                items={goalItems}
                selectedKey={route.goal}
                onSelectionChange={(key) => {
                  if (key)
                    updateRoute(route.id, { goal: String(key) as RoutingGoal });
                }}
              />
            </article>
          );
        })}
      </div>
    </section>
  );
}
