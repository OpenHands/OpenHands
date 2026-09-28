import { AgentServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";

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

function getClient(): AgentServerClient {
  return new AgentServerClient(getAgentServerClientOptions());
}

class ToolCatalogService {
  /** Tools this server offers, in the order it lists them. */
  static async getCatalog(): Promise<ToolCatalogEntry[]> {
    const response =
      await getClient().get<ToolCatalogResponse>(TOOL_CATALOG_PATH);
    return response?.tools ?? [];
  }
}

export default ToolCatalogService;
