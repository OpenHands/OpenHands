import React from "react";
import { useTranslation } from "react-i18next";
import { ROUTING_TARGET_AUTO } from "#/api/routing-service/routing-constants";
import type {
  RoutingPreset,
  RoutingResolveResult,
} from "#/api/routing-service/routing-types";
import { BenchmarkSources } from "#/components/features/routing/benchmark-sources";
import { DecisionTraceDrawer } from "#/components/features/routing/decision-trace-drawer";
import { DryRunConsole } from "#/components/features/routing/dry-run-console";
import { GoalGuardrails } from "#/components/features/routing/goal-guardrails";
import { HowAutoRoutes } from "#/components/features/routing/how-auto-routes";
import { ModelRegistryBrowser } from "#/components/features/routing/model-registry";
import { RouterModelChooser } from "#/components/features/routing/router-model-chooser";
import { RoutesTable } from "#/components/features/routing/routes-table";
import { TaxonomyEditor } from "#/components/features/routing/taxonomy-editor";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import {
  useIngestRoutingBenchmarks,
  usePutRoutingConfig,
  usePutRoutingRouterModel,
  usePutRoutingTaxonomy,
  useRoutingConfig,
  useRoutingLocalRuntimes,
  useRoutingRegistry,
  useRoutingResolve,
  useRoutingRouterModel,
  useRoutingSources,
  useRoutingTaxonomy,
} from "#/hooks/query/use-routing";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export function RoutingPage() {
  const { t } = useTranslation("openhands");
  const configQuery = useRoutingConfig();
  const taxonomyQuery = useRoutingTaxonomy();
  const registryQuery = useRoutingRegistry();
  const sourcesQuery = useRoutingSources();
  const routerModelQuery = useRoutingRouterModel();
  const runtimesQuery = useRoutingLocalRuntimes();
  const putConfig = usePutRoutingConfig();
  const putTaxonomy = usePutRoutingTaxonomy();
  const putRouterModel = usePutRoutingRouterModel();
  const ingest = useIngestRoutingBenchmarks();
  const resolve = useRoutingResolve();
  const [traceResult, setTraceResult] =
    React.useState<RoutingResolveResult | null>(null);
  const [previews, setPreviews] = React.useState<Record<string, string>>({});

  React.useEffect(() => {
    const routes = configQuery.data?.routes ?? [];
    let cancelled = false;
    (async () => {
      const next: Record<string, string> = {};
      await Promise.all(
        routes
          .filter((route) => route.target === ROUTING_TARGET_AUTO)
          .map(async (route) => {
            const result = await resolve.mutateAsync({
              task_text: route.id,
              work_type: route.work_type ?? undefined,
              sensitivity: route.sensitivity ?? undefined,
            });
            if (result.decision.provider_key && result.decision.model) {
              next[route.id] =
                `${result.decision.provider_key}/${result.decision.model}`;
            }
          }),
      );
      if (!cancelled) setPreviews(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [configQuery.data?.routes]);

  return (
    <div className="flex flex-col gap-8" data-testid="routing-page-body">
      {configQuery.data ? (
        <RoutesTable
          config={configQuery.data}
          previews={previews}
          onChange={(patch) => putConfig.mutate(patch)}
        />
      ) : null}
      {routerModelQuery.data ? (
        <RouterModelChooser
          routerModel={routerModelQuery.data}
          localRuntimes={runtimesQuery.data}
          onSelectPreset={(preset: RoutingPreset) =>
            putRouterModel.mutate({ preset })
          }
        />
      ) : null}
      {configQuery.data ? (
        <GoalGuardrails
          config={configQuery.data}
          onChange={(patch) => putConfig.mutate(patch)}
        />
      ) : null}
      {taxonomyQuery.data ? (
        <TaxonomyEditor
          taxonomy={taxonomyQuery.data}
          onChange={(next) => putTaxonomy.mutate(next)}
          onReset={() => putTaxonomy.mutate({ reset: true })}
        />
      ) : null}
      {sourcesQuery.data ? (
        <BenchmarkSources
          sources={sourcesQuery.data}
          ingesting={ingest.isPending}
          onIngest={(sourceId) => {
            ingest.mutate(sourceId ? [sourceId] : undefined, {
              onSuccess: (result) => {
                const failed = Object.values(result.sources).some(
                  (item) => !item.ok,
                );
                if (failed) {
                  displayErrorToast(t(I18nKey.ROUTING$INGEST_FAILED));
                }
              },
              onError: () =>
                displayErrorToast(t(I18nKey.ROUTING$INGEST_FAILED)),
            });
          }}
        />
      ) : null}
      {registryQuery.data ? (
        <ModelRegistryBrowser registry={registryQuery.data} />
      ) : null}
      <HowAutoRoutes />
      <DryRunConsole
        pending={resolve.isPending}
        result={traceResult}
        onResolve={(taskText) => {
          resolve.mutate(
            { task_text: taskText },
            {
              onSuccess: (result) => setTraceResult(result),
            },
          );
        }}
      />
      <DecisionTraceDrawer
        trace={traceResult?.trace ?? null}
        onClose={() => setTraceResult(null)}
      />
    </div>
  );
}

export function RoutingPageHeader() {
  const { t } = useTranslation("openhands");
  return <Typography.H2>{t(I18nKey.ROUTING$TITLE)}</Typography.H2>;
}
