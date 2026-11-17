import type { Backend } from "../backend-registry/types";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "./proxy";

export type WorklogContentType = "markdown" | "json";

export interface Worklog {
  id: string;
  factory_id: string;
  /** The identifier in URLs. It never changes; `name` can. */
  slug: string;
  name: string;
  is_default: boolean;
  created_at: string;
  updated_at: string;
  entry_count: number;
  json_entry_count: number;
  last_entry_at: string | null;
}

export interface WorklogEntryAuthor {
  kind: "user" | "automation";
  id: string | null;
  /** Set for automations; Canvas resolves a user's name from `id`. */
  name: string | null;
}

export interface WorklogEntry {
  id: number;
  created_at: string;
  content_type: WorklogContentType;
  title: string | null;
  author: WorklogEntryAuthor;
  run_id: string | null;
  body: string;
}

export interface WorklogEntryPage {
  items: WorklogEntry[];
  /** Position of the last entry returned; pass it back as `cursor`. */
  next_cursor: string | null;
  has_more: boolean;
}

export interface ListWorklogEntriesOptions {
  limit: number;
  cursor?: string | null;
  contentType?: WorklogContentType;
}

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Worklog calls require a cloud backend.");
  }
  return active;
}

function worklogsPath(orgId: string, factoryId: string): string {
  return `/api/organizations/${encodeURIComponent(orgId)}/factories/${encodeURIComponent(factoryId)}/worklogs`;
}

/** `GET .../worklogs`: a factory's worklogs with their entry counts. */
export async function listWorklogs(
  orgId: string,
  factoryId: string,
): Promise<Worklog[]> {
  const data = await callCloudProxy<{ items: Worklog[] }>({
    backend: getActiveCloudBackend(),
    method: "GET",
    path: worklogsPath(orgId, factoryId),
    orgId,
  });
  return data.items;
}

/**
 * `GET .../worklogs/{slug}/entries`, newest first. `cursor` is the previous
 * page's `next_cursor` and continues towards older entries.
 */
export async function listWorklogEntries(
  orgId: string,
  factoryId: string,
  worklogSlug: string,
  { limit, cursor, contentType }: ListWorklogEntriesOptions,
): Promise<WorklogEntryPage> {
  const query = new URLSearchParams({ order: "desc", limit: String(limit) });
  if (cursor) query.set("before", cursor);
  if (contentType) query.set("content_type", contentType);
  return callCloudProxy<WorklogEntryPage>({
    backend: getActiveCloudBackend(),
    method: "GET",
    path: `${worklogsPath(orgId, factoryId)}/${encodeURIComponent(worklogSlug)}/entries?${query}`,
    orgId,
  });
}
