import React from "react";
import { useTranslation } from "react-i18next";
import {
  GRAPH_QUERIES,
  GRAPH_QUERY_CALLERS,
  GRAPH_SOURCE,
} from "#/api/graph-service/graph-constants";
import type {
  GraphQueryKind,
  GraphQueryResult,
} from "#/api/graph-service/graph-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { formControlFieldClassName } from "#/utils/form-control-classes";
import { Typography } from "#/ui/typography";

const QUERY_LABELS: Record<GraphQueryKind, I18nKey> = {
  callers: I18nKey.GRAPH$QUERY_CALLERS,
  deps: I18nKey.GRAPH$QUERY_DEPS,
  usages: I18nKey.GRAPH$QUERY_USAGES,
};

export interface QueryConsoleProps {
  onQuery: (params: {
    q: GraphQueryKind;
    symbol: string;
    file: string;
  }) => void;
  result: GraphQueryResult | null;
  isBusy?: boolean;
}

export function QueryConsole({
  onQuery,
  result,
  isBusy = false,
}: QueryConsoleProps) {
  const { t } = useTranslation("openhands");
  const [kind, setKind] = React.useState<GraphQueryKind>(GRAPH_QUERY_CALLERS);
  const [symbol, setSymbol] = React.useState("");
  const [file, setFile] = React.useState("");

  return (
    <section
      data-testid="graph-query-console"
      className="rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.GRAPH$QUERY)}</Typography>
      <div className="mt-3 flex flex-col gap-2">
        <label className="text-xs text-tertiary-light">
          {t(I18nKey.GRAPH$QUERY)}
          <select
            data-testid="graph-query-kind"
            className={formControlFieldClassName}
            value={kind}
            onChange={(event) => setKind(event.target.value as GraphQueryKind)}
          >
            {GRAPH_QUERIES.map((item) => (
              <option key={item} value={item}>
                {t(QUERY_LABELS[item])}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-tertiary-light">
          {t(I18nKey.GRAPH$SYMBOL)}
          <input
            data-testid="graph-query-symbol"
            className={formControlFieldClassName}
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
          />
        </label>
        <label className="text-xs text-tertiary-light">
          {t(I18nKey.GRAPH$FILE)}
          <input
            data-testid="graph-query-file"
            className={formControlFieldClassName}
            value={file}
            onChange={(event) => setFile(event.target.value)}
          />
        </label>
        <BrandButton
          type="button"
          variant="primary"
          testId="graph-query-submit"
          isDisabled={isBusy}
          onClick={() => onQuery({ q: kind, symbol, file })}
        >
          {t(I18nKey.GRAPH$QUERY)}
        </BrandButton>
      </div>
      <p
        data-testid="graph-budget-notice"
        className="mt-2 text-xs text-tertiary-light"
      >
        {t(I18nKey.GRAPH$BUDGET)}
      </p>
      {result?.status === "stale" ? (
        <p
          data-testid="graph-stale-banner"
          className="mt-2 text-xs text-amber-400"
        >
          {t(I18nKey.GRAPH$STALE)}
        </p>
      ) : null}
      {result ? (
        <div className="mt-3 overflow-x-auto">
          <p
            data-testid="graph-query-source"
            className="text-xs text-tertiary-light"
          >
            {t(I18nKey.GRAPH$SOURCE)} {GRAPH_SOURCE}
          </p>
          <table
            data-testid="graph-query-results"
            className="mt-2 w-full text-sm"
          >
            <thead>
              <tr className="text-left text-tertiary-light">
                <th>{t(I18nKey.GRAPH$SYMBOL)}</th>
                <th>{t(I18nKey.GRAPH$KIND)}</th>
                <th>{t(I18nKey.GRAPH$FILE)}</th>
                <th>{t(I18nKey.GRAPH$LINE)}</th>
              </tr>
            </thead>
            <tbody>
              {result.result.length === 0 ? (
                <tr>
                  <td colSpan={4}>{t(I18nKey.GRAPH$NO_RESULTS)}</td>
                </tr>
              ) : (
                result.result.map((item) => (
                  <tr key={item.id} data-testid={`graph-result-${item.id}`}>
                    <td>{item.name}</td>
                    <td>{item.kind}</td>
                    <td>{item.path}</td>
                    <td>{item.start_line}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
