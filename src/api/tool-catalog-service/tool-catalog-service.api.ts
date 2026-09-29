import { AgentServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";

const TOOL_CATALOG_PATH = "/api/tools/catalog";

/** A tool as offered for configuring an agent. */
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
