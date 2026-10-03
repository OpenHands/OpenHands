import { AgentServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";

/** Well-known path of the agent-server tool catalog endpoint. */
export const TOOL_CATALOG_PATH = "/api/tools/catalog";

/**
 * A tool as offered for configuring an agent.
 *
 * @remarks Mirrors the agent-server's `ToolCatalogEntry` until
 * `@openhands/typescript-client` ships it.
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

function isToolCatalogEntry(value: unknown): value is ToolCatalogEntry {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.name === "string" &&
    typeof entry.user_selectable === "boolean" &&
    typeof entry.usable === "boolean" &&
    typeof entry.in_default_set === "boolean" &&
    (entry.description === undefined || typeof entry.description === "string")
  );
}

function assertValidCatalog(
  tools: unknown[],
): asserts tools is ToolCatalogEntry[] {
  for (const tool of tools) {
    if (!isToolCatalogEntry(tool)) {
      const name =
        typeof tool === "object" && tool !== null
          ? ((tool as Record<string, unknown>).name ?? "<missing>")
          : "<malformed>";
      throw new Error(
        `The agent server returned a malformed tool catalog entry ` +
          `(name="${String(name)}"); refusing to pick tools from it.`,
      );
    }
  }
}

class ToolCatalogService {
  /** Tools this server offers, in its order. */
  static async getCatalog(): Promise<ToolCatalogEntry[]> {
    const response = await fetchCatalog();
    const tools = response?.tools;
    if (!Array.isArray(tools)) {
      throw new Error(
        "The agent server returned a malformed tool catalog (missing the tools array).",
      );
    }
    assertValidCatalog(tools);
    return tools;
  }
}

export default ToolCatalogService;
