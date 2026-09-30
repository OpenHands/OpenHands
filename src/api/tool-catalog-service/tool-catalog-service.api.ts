import { AgentServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";

/** Well-known path of the agent-server tool catalog endpoint. */
export const TOOL_CATALOG_PATH = "/api/tools/catalog";

/**
 * A tool as offered for configuring an agent.
 *
 * @remarks Temporary local mirror of the agent-server's `ToolCatalogEntry`
 * (software-agent-sdk#5151). Per the repository contract order (Agent Server
 * contract → TypeScript client → Canvas), this belongs
 * in `@openhands/typescript-client`; it is declared here because the published
 * client (pinned 1.49.6) predates the unreleased endpoint and therefore
 * has no typed accessor for ityet. Once the SDK release carrying #5151 is
 * out, consume the client's catalog model and drop this interface —
 * see Reply to review on OpenHands/OpenHands#17516.
 */
export interface ToolCatalogEntry {
  name: string;
  /** Whether a user may pick this tool; false for built-ins and internals. */
  user_selectable: boolean;
  /** Whether this server's runtime can actually run it. */
  usable: boolean;
  /** Whether the tool belongs to the standard set a profile gets by default. */
  in_default_set: boolean;
  description?: string;
}

interface ToolCatalogResponse {
  tools: ToolCatalogEntry[];
}

function fetchCatalog(): Promise<ToolCatalogResponse> {
  const { backend } = getActiveBackend();
  if (backend.kind === "cloud") {
    return callCloudProxy<ToolCatalogResponse>({
      backend,
      method: "GET",
      path: TOOL_CATALOG_PATH,
    });
  }
  return new AgentServerClient(getAgentServerClientOptions()).get(
    TOOL_CATALOG_PATH,
  );
}

class ToolCatalogService {
  /** Tools this server offers, in its order. */
  static async getCatalog(): Promise<ToolCatalogEntry[]> {
    const response = await fetchCatalog();
    return response?.tools ?? [];
  }
}

export default ToolCatalogService;
