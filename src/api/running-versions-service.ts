import { ServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";

export interface BackendRunningVersions {
  serviceVersion: string | null;
  sdkVersion: string | null;
  toolsVersion: string | null;
  workspaceVersion: string | null;
}

interface ServerInfoResponse {
  version?: unknown;
  app_version?: unknown;
  sdk_version?: unknown;
  tools_version?: unknown;
  workspace_version?: unknown;
}

function normalizeVersion(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const version = value.trim();
  return version && version.toLowerCase() !== "unknown" ? version : null;
}

function normalizeServerInfo(
  info: ServerInfoResponse,
): BackendRunningVersions | null {
  const versions = {
    serviceVersion: normalizeVersion(info.version ?? info.app_version),
    sdkVersion: normalizeVersion(info.sdk_version),
    toolsVersion: normalizeVersion(info.tools_version),
    workspaceVersion: normalizeVersion(info.workspace_version),
  };
  return Object.values(versions).some(Boolean) ? versions : null;
}

class RunningVersionsService {
  static async getBackendVersions(
    active: ResolvedActiveBackend = getActiveBackend(),
  ): Promise<BackendRunningVersions | null> {
    try {
      let info: ServerInfoResponse;
      if (active.backend.kind === "cloud") {
        info = await callCloudProxy<ServerInfoResponse>({
          backend: active.backend,
          method: "GET",
          path: "/server_info",
          headers: active.orgId ? { "X-Org-Id": active.orgId } : undefined,
          timeoutSeconds: 5,
        });
      } else {
        const options = getAgentServerClientOptions({
          host: active.backend.host,
          sessionApiKey: active.backend.apiKey,
          timeout: 5000,
        });
        info = await new ServerClient(options).getServerInfo();
      }

      return normalizeServerInfo(info);
    } catch {
      return null;
    }
  }
}

export default RunningVersionsService;
