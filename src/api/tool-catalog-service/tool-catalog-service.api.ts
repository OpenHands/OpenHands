/**
 * The tools an agent-server offers for configuring an agent, and what a draft
 * profile resolves to.
 *
 * Both endpoints shipped with the tool catalog (software-agent-sdk#4958) and
 * are absent from the pinned ts-client, so they go through the generic
 * `AgentServerClient`. Call sites gate on `agentProfileSupportsToolCatalog()`
 * rather than probing: a server without the catalog simply offers no picker.
 */
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
}

interface ToolCatalogResponse {
  tools: ToolCatalogEntry[];
}

/** A profile body evaluated without saving it. */
export type DraftAgentProfile = Record<string, unknown> & { name: string };

interface MaterializeResponse {
  resolved_settings?: { tools?: { name?: unknown }[] | null } | null;
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

  /**
   * Tool names a draft profile would launch with.
   *
   * Asks the server rather than reproducing its defaulting rules, so the editor
   * shows the browser tool exactly where the runtime can run it.
   */
  static async getResolvedToolNames(
    profile: DraftAgentProfile,
  ): Promise<string[]> {
    const response = await getClient().post<MaterializeResponse>(
      `/api/agent-profiles/${encodeURIComponent(profile.name)}/materialize`,
      { profile },
    );
    const tools = response?.resolved_settings?.tools ?? [];
    return tools
      .map(({ name }) => name)
      .filter((name): name is string => typeof name === "string");
  }
}

export default ToolCatalogService;
