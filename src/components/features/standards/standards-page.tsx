import React from "react";
import { useTranslation } from "react-i18next";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";
import type {
  StandardsConfig,
  StandardsPluginInfo,
  StandardsRunResult,
} from "#/api/standards-service/standards-types";
import { StandardsAuditLog } from "#/components/features/standards/standards-audit-log";
import { StandardsConsole } from "#/components/features/standards/standards-console";
import { StandardsEnforcement } from "#/components/features/standards/standards-enforcement";
import { StandardsPluginGallery } from "#/components/features/standards/standards-plugin-gallery";
import {
  usePutStandardsConfig,
  useRunStandards,
  useStandardsAudit,
  useStandardsConfig,
  useStandardsPlugins,
} from "#/hooks/query/use-standards";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export function StandardsPage() {
  const { t } = useTranslation("openhands");
  const [root, setRoot] = React.useState(DEFAULT_WORKING_DIR);
  const [result, setResult] = React.useState<StandardsRunResult | null>(null);
  const [pluginFilter, setPluginFilter] = React.useState("");
  const [severityFilter, setSeverityFilter] = React.useState("");
  const [fileFilter, setFileFilter] = React.useState("");
  const [beforeId, setBeforeId] = React.useState<string | undefined>();

  const pluginsQuery = useStandardsPlugins(root);
  const configQuery = useStandardsConfig(root);
  const putConfig = usePutStandardsConfig();
  const runChecks = useRunStandards();
  const auditQuery = useStandardsAudit({
    plugin: pluginFilter || undefined,
    severity: severityFilter || undefined,
    file: fileFilter || undefined,
    before_id: beforeId,
  });

  const plugins = pluginsQuery.data?.plugins ?? [];
  const config = configQuery.data;

  const persist = (patch: Partial<StandardsConfig>) => {
    if (!config) return;
    putConfig.mutate({
      ...config,
      ...patch,
      enforcement: patch.enforcement
        ? { ...config.enforcement, ...patch.enforcement }
        : config.enforcement,
      plugins: patch.plugins ?? config.plugins,
      root,
    });
  };

  const onToggle = (plugin: StandardsPluginInfo) => {
    if (!config) return;
    const existing = config.plugins.find((item) => item.name === plugin.name);
    const next = existing
      ? config.plugins.map((item) =>
          item.name === plugin.name
            ? { ...item, enabled: !item.enabled }
            : item,
        )
      : [
          ...config.plugins,
          {
            name: plugin.name,
            enabled: !plugin.enabled,
            action: plugin.action,
          },
        ];
    persist({ plugins: next });
  };

  return (
    <div
      data-testid="standards-page"
      className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6"
    >
      <div>
        <Typography variant="h2">{t(I18nKey.STANDARDS$TITLE)}</Typography>
        <p className="mt-1 text-sm text-tertiary-light">
          {t(I18nKey.STANDARDS$SUBTITLE)}
        </p>
      </div>
      <StandardsPluginGallery plugins={plugins} onToggle={onToggle} />
      {config ? (
        <StandardsEnforcement
          config={config}
          plugins={plugins}
          onChange={persist}
        />
      ) : null}
      <StandardsConsole
        root={root}
        onRootChange={setRoot}
        onRun={() => {
          runChecks.mutate(
            { root },
            { onSuccess: (payload) => setResult(payload) },
          );
        }}
        isBusy={runChecks.isPending}
        result={result}
      />
      <StandardsAuditLog
        page={auditQuery.data}
        plugin={pluginFilter}
        severity={severityFilter}
        file={fileFilter}
        onPluginChange={(value) => {
          setBeforeId(undefined);
          setPluginFilter(value);
        }}
        onSeverityChange={(value) => {
          setBeforeId(undefined);
          setSeverityFilter(value);
        }}
        onFileChange={(value) => {
          setBeforeId(undefined);
          setFileFilter(value);
        }}
        onLoadOlder={() => {
          if (auditQuery.data?.next_before_id) {
            setBeforeId(auditQuery.data.next_before_id);
          }
        }}
      />
    </div>
  );
}
