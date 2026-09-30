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
 * in `@openhands/typescript-client`. The published client cannot be consumed
 * yet: software-agent-sdk#5151 is still open as of 2026-09-30,and the
 * npm `@openhands/typescript-client` (1.50.1) ships no catalog type for
 * the endpoint. Once the SDK release carrying #5151 is out, consume the
 * client's catalog modeland drop this interface(see Reply to review on
 * OpenHands/OpenHands#17516). Until then,the shape guard in
 * {@link ToolCatalogService.getCatalog} keeps a catalog entry that omits one
 * of the discriminator fields from silently truncating the picker: it fails
 * loudly instead. The fields are required on the wire;the server's pydantic
 * model defaults them, so a response always carries them;a variant that
 * omits one is malformed, not nullable.
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
