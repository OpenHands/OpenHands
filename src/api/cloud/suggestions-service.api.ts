import { SuggestedTask } from "#/utils/types";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud suggested-tasks call requires a cloud backend.");
  }
  return active;
}

export async function getCloudSuggestedTasks(args: {
  pageId?: string;
  limit?: number;
}): Promise<{ items: SuggestedTask[]; next_page_id: string | null }> {
  const backend = getActiveCloudBackend();
  const params = new URLSearchParams();
  params.set("limit", String(args.limit ?? 30));
  if (args.pageId) params.set("page_id", args.pageId);

  const data = await callCloudProxy<{
    items: SuggestedTask[];
    next_page_id: string | null;
  }>({
    backend,
    method: "GET",
    path: `/api/v1/git/suggested-tasks/search?${params.toString()}`,
  });

  return {
    items: data?.items ?? [],
    next_page_id: data?.next_page_id ?? null,
  };
}
