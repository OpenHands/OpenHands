import { useTranslation } from "react-i18next";
import type { GraphRelevantFile } from "#/api/graph-service/graph-types";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { Typography } from "#/ui/typography";

export interface RelevantFilesCardProps {
  files: GraphRelevantFile[];
  budgetLines: number;
  usedLines: number;
  enabled: boolean;
  defaultEnabled: boolean;
  staleSkipped?: boolean;
  onToggle: () => void;
}

export function RelevantFilesCard({
  files,
  budgetLines,
  usedLines,
  enabled,
  defaultEnabled,
  staleSkipped = false,
  onToggle,
}: RelevantFilesCardProps) {
  const { t } = useTranslation("openhands");

  return (
    <section
      data-testid="graph-relevant-files"
      className="flex flex-col gap-3 rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.GRAPH$RELEVANT)}</Typography>
      <ToggleSwitch
        enabled={enabled}
        label={t(I18nKey.GRAPH$CONTEXT_TOGGLE)}
        onToggle={onToggle}
      />
      <p
        data-testid="graph-relevant-default"
        data-default-enabled={String(defaultEnabled)}
        className="sr-only"
      >
        {t(I18nKey.GRAPH$ENABLED)}
      </p>
      <p data-testid="graph-budget-usage" className="text-sm text-white">
        {t(I18nKey.GRAPH$BUDGET_USAGE)} {usedLines}/{budgetLines}
      </p>
      {staleSkipped ? (
        <p data-testid="graph-stale-skipped" className="text-sm text-amber-400">
          {t(I18nKey.GRAPH$STALE_SKIPPED)}
        </p>
      ) : null}
      <table className="w-full text-left text-sm text-white">
        <thead>
          <tr className="text-tertiary-light">
            <th>{t(I18nKey.GRAPH$FILE)}</th>
            <th>{t(I18nKey.GRAPH$REASON)}</th>
          </tr>
        </thead>
        <tbody>
          {files.length === 0 ? (
            <tr>
              <td colSpan={2}>{t(I18nKey.GRAPH$NO_RESULTS)}</td>
            </tr>
          ) : (
            files.map((item) => (
              <tr
                key={item.file}
                data-testid={`graph-relevant-row-${item.file}`}
              >
                <td>{item.file}</td>
                <td>{item.reason}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </section>
  );
}
