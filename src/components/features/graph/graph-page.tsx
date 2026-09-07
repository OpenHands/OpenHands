import React from "react";
import { useTranslation } from "react-i18next";
import { GraphSettings } from "#/components/features/graph/graph-settings";
import { IndexerStatusCard } from "#/components/features/graph/indexer-status-card";
import {
  useClearGraphIndex,
  useGraphConfig,
  useImportGraphProjectConfig,
  usePutGraphConfig,
  useRetriggerGraphIndex,
  useGraphStatus,
} from "#/hooks/query/use-graph";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export function GraphIndexerControls() {
  const { t } = useTranslation("openhands");
  const statusQuery = useGraphStatus();
  const configQuery = useGraphConfig();
  const putConfig = usePutGraphConfig();
  const clearIndex = useClearGraphIndex();
  const retrigger = useRetriggerGraphIndex();
  const importConfig = useImportGraphProjectConfig();

  const status = statusQuery.data;
  const config = configQuery.data;
  if (!status && !config) return null;

  return (
    <section
      data-testid="graph-indexer-controls"
      className="flex flex-col gap-4"
    >
      <div>
        <Typography variant="h3">{t(I18nKey.GRAPH$TITLE)}</Typography>
        <p className="mt-1 text-sm text-tertiary-light">
          {t(I18nKey.GRAPH$AGENT_INDEX_HINT)}
        </p>
      </div>
      {status ? (
        <IndexerStatusCard
          status={status}
          isBusy={clearIndex.isPending || retrigger.isPending}
          onClear={() => clearIndex.mutate(undefined)}
          onRetrigger={() => retrigger.mutate(undefined)}
        />
      ) : null}
      {config ? (
        <GraphSettings
          config={config}
          onChange={(patch) => putConfig.mutate(patch)}
          onImport={(path) => importConfig.mutate(path)}
        />
      ) : null}
    </section>
  );
}

export const GraphPage = GraphIndexerControls;
