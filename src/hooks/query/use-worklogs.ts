import { useEffect, useRef } from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import {
  listWorklogEntries,
  listWorklogs,
  type Worklog,
} from "#/api/cloud/worklog-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudLicenseStatus } from "./use-cloud-license-status";
import { useFactoriesSupported } from "./use-factories";

export const WORKLOGS_QUERY_KEY = ["worklogs"] as const;

/** How often the worklog list is re-read to notice new entries. */
export const WORKLOG_POLL_INTERVAL_MS = 5_000;
const WORKLOG_ERROR_POLL_INTERVAL_MS = 30_000;

export const WORKLOG_ENTRIES_PAGE_SIZE = 20;

/**
 * Whether the worklog UI applies: the automation service serves factories and
 * the Enterprise instance's license grants `worklogs`. A license status that is
 * missing, lacks the key (an instance that predates worklogs) or fails to load
 * all count as not granted.
 */
export function useWorklogsSupported() {
  const factories = useFactoriesSupported();
  const license = useCloudLicenseStatus();
  return {
    isLoading: factories.isLoading || license.isLoading,
    isSupported:
      factories.isSupported && license.data?.features?.worklogs === true,
  };
}

/** A factory's worklogs, re-read every few seconds. */
export function useWorklogs(factoryId: string, { enabled = true } = {}) {
  const { backend, orgId } = useActiveBackend();

  return useQuery({
    queryKey: [
      ...WORKLOGS_QUERY_KEY,
      "list",
      backend.id,
      orgId,
      factoryId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => listWorklogs(orgId!, factoryId),
    enabled: enabled && !!orgId && factoryId !== "",
    refetchInterval: (query) =>
      query.state.status === "error"
        ? WORKLOG_ERROR_POLL_INTERVAL_MS
        : WORKLOG_POLL_INTERVAL_MS,
    staleTime: 0,
    retry: false,
    meta: { disableToast: true },
  });
}

/**
 * One worklog's entries, newest first, a page at a time. Pass the worklog as
 * last read by {@link useWorklogs}: when its entry count or last entry time
 * changes, the loaded pages are re-read from the newest end. Mount it per
 * worklog (`key` the slug) so a switch does not read the previous stamp.
 */
export function useWorklogEntries(
  factoryId: string,
  worklog: Worklog,
  { includeJson }: { includeJson: boolean },
) {
  const { backend, orgId } = useActiveBackend();
  const { slug } = worklog;

  const query = useInfiniteQuery({
    queryKey: [
      ...WORKLOGS_QUERY_KEY,
      "entries",
      backend.id,
      orgId,
      factoryId,
      slug,
      includeJson,
      backend.connectionRevision ?? 0,
    ],
    queryFn: ({ pageParam }) =>
      listWorklogEntries(orgId!, factoryId, slug, {
        limit: WORKLOG_ENTRIES_PAGE_SIZE,
        cursor: pageParam,
        contentType: includeJson ? undefined : "markdown",
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) =>
      lastPage.has_more ? lastPage.next_cursor : undefined,
    enabled: !!orgId,
    staleTime: Infinity,
    retry: false,
    meta: { disableToast: true },
  });

  const stamp = `${worklog.entry_count}:${worklog.json_entry_count}:${worklog.last_entry_at}`;
  const seenStamp = useRef(stamp);
  const { refetch } = query;
  useEffect(() => {
    if (seenStamp.current === stamp) return;
    seenStamp.current = stamp;
    void refetch();
  }, [stamp, refetch]);

  return query;
}
