import React from "react";
import { useTranslation } from "react-i18next";
import type { GraphQueryKind } from "#/api/graph-service/graph-types";
import { GraphSettings } from "#/components/features/graph/graph-settings";
import { IndexerStatusCard } from "#/components/features/graph/indexer-status-card";
import { QueryConsole } from "#/components/features/graph/query-console";
import { RelevantFilesCard } from "#/components/features/graph/relevant-files-card";
import {
  useClearGraphIndex,
  useGraphConfig,
  useGraphQuery,
  useGraphStatus,
  useImportGraphProjectConfig,
  usePutGraphConfig,
  useRetriggerGraphIndex,
} from "#/hooks/query/use-graph";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export function GraphPageHeader() {
  const { t } = useTranslation("openhands");
  return (
    <Typography variant="h2" testId="graph-title">
      {t(I18nKey.GRAPH$TITLE)}
    </Typography>
  );
}

export function GraphPage() {
  const statusQuery = useGraphStatus();
  const configQuery = useGraphConfig();
  const putConfig = usePutGraphConfig();
  const clearIndex = useClearGraphIndex();
  const retrigger = useRetriggerGraphIndex();
  const query = useGraphQuery();
  const importConfig = useImportGraphProjectConfig();
  const [runEnabled, setRunEnabled] = React.useState<boolean | null>(null);

  const status = statusQuery.data;
  const config = configQuery.data;
  const defaultEnabled = config?.enabled ?? true;
  const enabled = runEnabled ?? defaultEnabled;

  return (
    <div className="flex flex-col gap-4 pb-8">
      {status ? (
        <IndexerStatusCard
          status={status}
          isBusy={clearIndex.isPending || retrigger.isPending}
          onClear={() => clearIndex.mutate(undefined)}
          onRetrigger={() => retrigger.mutate(undefined)}
        />
      ) : null}
      {config ? (
        <RelevantFilesCard
          files={[]}
          budgetLines={config.graph_budget_lines}
          usedLines={0}
          enabled={enabled}
          defaultEnabled={defaultEnabled}
          onToggle={() => setRunEnabled(!enabled)}
        />
      ) : null}
      <QueryConsole
        isBusy={query.isPending}
        result={query.data ?? null}
        onQuery={(params) =>
          query.mutate({
            q: params.q as GraphQueryKind,
            symbol: params.symbol || undefined,
            file: params.file || undefined,
          })
        }
      />
      {config ? (
        <GraphSettings
          config={config}
          onChange={(patch) => putConfig.mutate(patch)}
          onImport={(path) => importConfig.mutate(path)}
        />
      ) : null}
    </div>
  );
}
