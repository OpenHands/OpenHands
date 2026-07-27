import { PluginsClient } from "@openhands/typescript-client/clients";
import { getActiveBackend } from "./backend-registry/active-store";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import type { PluginBundledSkill } from "./plugins-service";

/**
 * An installed plugin, as returned by the agent-server management router
 * (`GET /api/plugins/installed`). Matches the typescript-client
 * `InstalledPluginInfo`. The contents fields (`skills`, `files`, relative to
 * `install_path`) are absent on older agent-servers and null when the
 * installed plugin directory failed to load.
 */
export interface InstalledPluginInfo {
  name: string;
  version: string;
  description: string | null;
  enabled: boolean;
  source: string;
  resolved_ref?: string | null;
  repo_path?: string | null;
  installed_at: string;
  install_path: string;
  skills?: PluginBundledSkill[] | null;
  files?: string[] | null;
}

/** Coordinates for installing a plugin from a git source or local path. */
export interface InstallPluginRequest {
  source: string;
  ref?: string | null;
  repo_path?: string | null;
  force?: boolean;
}

/**
 * The slice of the typescript-client `PluginsClient` this service drives. The
 * installed `@openhands/typescript-client` package does not yet export these
 * management methods (they ship in typescript-client PRs #222/#223); narrowing
 * to this local interface keeps `vitest` (types stripped) and `eslint` green
 * until the client is republished, at which point the cast below can be dropped.
 */
interface PluginsManagementClient {
  listInstalledPlugins(): Promise<{ plugins: InstalledPluginInfo[] }>;
  installPlugin(request: InstallPluginRequest): Promise<InstalledPluginInfo>;
  setPluginEnabled(
    name: string,
    enabled: boolean,
  ): Promise<{ name: string; enabled: boolean }>;
  uninstallPlugin(name: string): Promise<{ message: string }>;
  refreshPlugin(
    name: string,
  ): Promise<{ message: string; plugin: InstalledPluginInfo }>;
}

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

function getManagementClient(): PluginsManagementClient {
  return new PluginsClient(
    getAgentServerClientOptions(),
  ) as unknown as PluginsManagementClient;
}

/**
 * Front-end management layer for installed plugins: list / install / enable /
 * disable / uninstall / refresh. Kept separate from the read-only catalog
 * service (`plugins-service.ts`), exactly as skills separate the marketplace
 * catalog from install actions.
 *
 * Local backend only for now (per Appendix C Q5): installed plugins live on the
 * local agent-server's `~/.openhands/plugins/installed/`. A cloud backend has no
 * per-user installed store yet, so reads return an empty list and mutating
 * actions throw (the UI also disables them on cloud).
 */
class PluginsManagementService {
  static async listInstalledPlugins(): Promise<InstalledPluginInfo[]> {
    if (isCloudBackend()) {
      return [];
    }

    try {
      const response = await getManagementClient().listInstalledPlugins();
      return response.plugins ?? [];
    } catch {
      // Agent-server may predate the plugins router or be unreachable; surface
      // an empty list rather than throwing (mirrors the catalog service).
      return [];
    }
  }

  static async installPlugin(
    request: InstallPluginRequest,
  ): Promise<InstalledPluginInfo> {
    if (isCloudBackend()) {
      throw new Error(
        "Installing plugins is only available on a local backend.",
      );
    }
    return getManagementClient().installPlugin(request);
  }

  static async setPluginEnabled(
    name: string,
    enabled: boolean,
  ): Promise<{ name: string; enabled: boolean }> {
    if (isCloudBackend()) {
      throw new Error(
        "Enabling and disabling plugins is only available on a local backend.",
      );
    }
    return getManagementClient().setPluginEnabled(name, enabled);
  }

  static async uninstallPlugin(name: string): Promise<{ message: string }> {
    if (isCloudBackend()) {
      throw new Error(
        "Uninstalling plugins is only available on a local backend.",
      );
    }
    return getManagementClient().uninstallPlugin(name);
  }

  static async refreshPlugin(
    name: string,
  ): Promise<{ message: string; plugin: InstalledPluginInfo }> {
    if (isCloudBackend()) {
      throw new Error(
        "Refreshing plugins is only available on a local backend.",
      );
    }
    return getManagementClient().refreshPlugin(name);
  }
}

export default PluginsManagementService;
