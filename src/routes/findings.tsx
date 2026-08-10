/**
 * Route: /findings — owns URL search params; components stay router-free.
 * @spec PROJETOSIN-188 — findings route
 */

import React from "react";
import { useSearchParams } from "react-router";
import {
  EMPTY_FINDINGS_FILTERS,
  FindingsPage,
  parseSeveritiesParam,
  parseStatusesParam,
} from "#/components/features/findings/findings-page";
import type { FindingsFilterState } from "#/components/features/findings/findings-filters";

function readFilters(params: URLSearchParams): FindingsFilterState {
  return {
    severities: parseSeveritiesParam(params.get("severity")),
    statuses: parseStatusesParam(params.get("status")),
    sourceTool: params.get("source_tool") ?? "",
    asset: params.get("asset") ?? "",
    titleQuery: params.get("q") ?? "",
  };
}

function writeFilters(
  previous: URLSearchParams,
  filters: FindingsFilterState,
  page: number,
  newOnly: boolean,
): URLSearchParams {
  const next = new URLSearchParams(previous);

  if (filters.severities.length > 0) {
    next.set("severity", filters.severities.join(","));
  } else {
    next.delete("severity");
  }

  if (newOnly) {
    next.set("status", "new");
  } else if (filters.statuses.length > 0) {
    next.set("status", filters.statuses.join(","));
  } else {
    next.delete("status");
  }

  if (filters.sourceTool.trim()) {
    next.set("source_tool", filters.sourceTool.trim());
  } else {
    next.delete("source_tool");
  }

  if (filters.asset.trim()) {
    next.set("asset", filters.asset.trim());
  } else {
    next.delete("asset");
  }

  if (filters.titleQuery.trim()) {
    next.set("q", filters.titleQuery.trim());
  } else {
    next.delete("q");
  }

  if (page > 1) {
    next.set("page", String(page));
  } else {
    next.delete("page");
  }

  return next;
}

export default function FindingsRoute() {
  const [searchParams, setSearchParams] = useSearchParams();

  const engagementId = searchParams.get("engagement_id");
  const page = Math.max(1, Number(searchParams.get("page") ?? "1") || 1);
  const filters = React.useMemo(
    () => readFilters(searchParams),
    [searchParams],
  );
  const newOnly =
    filters.statuses.length === 1 && filters.statuses[0] === "new";

  const updateParams = React.useCallback(
    (mutator: (params: URLSearchParams) => URLSearchParams) => {
      setSearchParams((previous) => mutator(new URLSearchParams(previous)), {
        replace: true,
      });
    },
    [setSearchParams],
  );

  return (
    <FindingsPage
      engagementId={engagementId}
      page={page}
      filters={filters}
      newOnly={newOnly}
      onFiltersChange={(next) => {
        updateParams((previous) => writeFilters(previous, next, 1, false));
      }}
      onClearFilters={() => {
        updateParams((previous) =>
          writeFilters(previous, EMPTY_FINDINGS_FILTERS, 1, false),
        );
      }}
      onToggleNewOnly={() => {
        updateParams((previous) => {
          if (newOnly) {
            return writeFilters(
              previous,
              { ...filters, statuses: [] },
              1,
              false,
            );
          }
          return writeFilters(
            previous,
            { ...filters, statuses: ["new"] },
            1,
            true,
          );
        });
      }}
      onPageChange={(nextPage) => {
        updateParams((previous) =>
          writeFilters(previous, filters, nextPage, newOnly),
        );
      }}
    />
  );
}
