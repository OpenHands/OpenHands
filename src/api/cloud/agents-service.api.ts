import type { AgentInfo } from "#/types/settings";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

interface CloudAgentsPage {
  items: AgentInfo[];
  next_page_id: string | null;
}

const PAGE_LIMIT = 100;

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud agents call requires a cloud backend.");
  }
  return active;
}

/**
 * Fetch the full list of agents from the cloud backend. The cloud endpoint is
 * paginated (page_id cursor); we walk all pages so the settings UI gets a
 * complete list in one call. The cloud AgentInfo shape matches the GUI's
 * AgentInfo type, so items are passed through unchanged. Mirrors
 * `fetchCloudSkills()` against the `GET /api/v1/agents/search` route.
 */
export async function fetchCloudAgents(): Promise<AgentInfo[]> {
  const backend = getActiveCloudBackend();

  const agents: AgentInfo[] = [];
  let pageId: string | null = null;

  do {
    const query = new URLSearchParams({ limit: String(PAGE_LIMIT) });
    if (pageId) query.set("page_id", pageId);

    const page = await callCloudProxy<CloudAgentsPage>({
      backend,
      method: "GET",
      path: `/api/v1/agents/search?${query.toString()}`,
    });

    agents.push(...(page.items ?? []));
    pageId = page.next_page_id;
  } while (pageId);

  return agents;
}
