import {
  FileClient,
  PluginsClient,
} from "@openhands/typescript-client/clients";
import { getActiveBackend } from "./backend-registry/active-store";
import { getAgentServerClientOptions } from "./agent-server-client-options";

/** Summary of a skill bundled in a plugin (agent-server `PluginSkillSummary`). */
export interface PluginBundledSkill {
  name: string;
  description?: string | null;
}

/**
 * A plugin in the dynamic marketplace catalog, with attachable coordinates and
 * install state. Matches the agent-server `MarketplacePluginInfo` / the
 * typescript-client `MarketplacePlugin`. The contents fields (`path`, `skills`,
 * `files`) are populated when the entry resolves to a directory in the
 * server's local marketplace clone, and are absent on older agent-servers.
 */
export interface MarketplacePlugin {
  name: string;
  description: string | null;
  source: string;
  ref?: string | null;
  repo_path?: string | null;
  installed: boolean;
  path?: string | null;
  skills?: PluginBundledSkill[] | null;
  files?: string[] | null;
}

/**
 * A locally-discovered ("ambient") plugin reported by the agent-server — one
 * found in the user's local plugin directories (e.g. `~/.agents/plugins`).
 * These auto-load into conversations and are not managed via install/uninstall,
 * so the Plugins page renders them as a read-only "Local" group. Matches the
 * typescript-client `PluginInfo`; the contents fields are absent on older
 * agent-servers.
 */
export interface LocalPlugin {
  name: string;
  version: string;
  description: string;
  path?: string;
  skills?: PluginBundledSkill[];
  files?: string[];
}

/** Content of a single plugin file fetched for the detail-modal viewer. */
export interface PluginFileContent {
  kind: "text" | "binary";
  text: string | null;
}

function isLikelyBinary(buffer: ArrayBuffer): boolean {
  // Same heuristic git uses: presence of a NUL byte in the first ~8KB. Small
  // private copy of `isLikelyBinary` in `use-workspace-file-content.ts` — that
  // module is conversation-workspace-specific and heavy to import from here.
  const view = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 8000));
  for (let i = 0; i < view.length; i += 1) {
    if (view[i] === 0) return true;
  }
  return false;
}

class PluginsService {
  /**
   * Fetch the dynamic plugins marketplace catalog.
   *
   * Local backend only for now: the catalog is fetched at run time from the
   * agent-server via the typed client (no bundled catalog, so the list stays
   * dynamic). On a cloud backend an empty catalog is returned — there is no
   * cloud plugins-marketplace endpoint yet (tracked as a follow-up ticket).
   */
  static async getPluginsMarketplace(): Promise<MarketplacePlugin[]> {
    if (getActiveBackend().backend.kind === "cloud") {
      return [];
    }

    try {
      const response = await new PluginsClient(
        getAgentServerClientOptions(),
      ).getPluginsMarketplace();
      return (response.plugins ?? []) as MarketplacePlugin[];
    } catch {
      // Agent-server may not support the plugins endpoint or be unreachable;
      // surface an empty catalog rather than throwing.
      return [];
    }
  }

  /**
   * Fetch the locally-discovered ("ambient") plugins from the agent-server.
   *
   * Only user-level plugins are requested (`~/.agents/plugins`,
   * `~/.openhands/plugins`, plus enabled installed plugins): the Plugins page is
   * global, so there is no project workspace to scope project plugins to.
   *
   * Local backend only — a cloud backend has no local plugin directories, so an
   * empty list is returned. Errors surface as an empty list (mirrors the
   * catalog) rather than throwing.
   */
  static async getLocalPlugins(): Promise<LocalPlugin[]> {
    if (getActiveBackend().backend.kind === "cloud") {
      return [];
    }

    try {
      const response = await new PluginsClient(
        getAgentServerClientOptions(),
      ).getPlugins({ load_user: true, load_project: false });
      return (response.plugins ?? []) as LocalPlugin[];
    } catch {
      return [];
    }
  }

  /**
   * Fetch one plugin file's content for the detail-modal viewer. `basePath` is
   * the plugin directory reported by the agent-server (`path`/`install_path`)
   * and `relativePath` a POSIX path from the plugin's `files` listing.
   *
   * Local backend only — plugin files live on the local agent-server's disk.
   * Errors propagate so the caller can render a load-error state.
   */
  static async getPluginFileContent(
    basePath: string,
    relativePath: string,
  ): Promise<PluginFileContent> {
    if (getActiveBackend().backend.kind === "cloud") {
      throw new Error(
        "Reading plugin files is only available on a local backend.",
      );
    }

    const buffer = await new FileClient(
      getAgentServerClientOptions(),
    ).downloadFile(`${basePath}/${relativePath}`);
    if (isLikelyBinary(buffer)) {
      return { kind: "binary", text: null };
    }
    return {
      kind: "text",
      text: new TextDecoder("utf-8", { fatal: false }).decode(buffer),
    };
  }
}

export default PluginsService;
