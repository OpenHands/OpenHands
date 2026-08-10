import axios, { type AxiosRequestConfig } from "axios";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { getEffectiveLocalBackend } from "#/api/backend-registry/active-store";
import { APPWRITE_WORKSPACE_ID_HEADER } from "#/utils/appwrite-integration-secrets";
import type { DependencyTrackFinding } from "#/utils/dependency-track-findings";
import { encodeBomForDependencyTrack } from "#/utils/syft-output";

export const DEPENDENCY_TRACK_PROXY_BASE = "/api/integrations/dependency-track";

const BOM_POLL_INTERVAL_MS = 2000;
const BOM_POLL_MAX_ATTEMPTS = 90;

async function dependencyTrackRequest<T>(
  workspaceId: string,
  method: string,
  path: string,
  options: {
    data?: unknown;
    params?: Record<string, string | number | boolean | undefined>;
    headers?: Record<string, string>;
    responseType?: AxiosRequestConfig["responseType"];
  } = {},
): Promise<T> {
  if (!workspaceId.trim()) {
    throw new Error("Dependency-Track workspace id is required");
  }
  const backend = getEffectiveLocalBackend();
  if (!backend) {
    throw new NoBackendAvailableError();
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${backend.host.replace(/\/+$/, "")}${DEPENDENCY_TRACK_PROXY_BASE}${normalizedPath}`;
  const apiKey = backend.apiKey?.trim();

  const response = await axios.request<T>({
    method,
    url,
    data: options.data,
    params: options.params,
    responseType: options.responseType,
    headers: {
      Accept: "application/json",
      [APPWRITE_WORKSPACE_ID_HEADER]: workspaceId,
      ...(apiKey ? { "X-Session-API-Key": apiKey } : {}),
      ...options.headers,
    },
  });
  return response.data;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export type DependencyTrackClient = ReturnType<
  typeof DependencyTrackService.forWorkspace
>;

export class DependencyTrackService {
  static forWorkspace(workspaceId: string) {
    const id = workspaceId.trim();
    const request = <T>(
      method: string,
      path: string,
      options?: Parameters<typeof dependencyTrackRequest<T>>[3],
    ) => dependencyTrackRequest<T>(id, method, path, options);

    return {
      async testConnection(): Promise<void> {
        await request("GET", "/api/v1/version");
      },

      async uploadBom(bomJson: string, projectUuid: string): Promise<string> {
        const response = await request<{ token?: string }>(
          "PUT",
          "/api/v1/bom",
          {
            data: {
              project: projectUuid,
              bom: encodeBomForDependencyTrack(bomJson),
            },
          },
        );
        const token = response.token?.trim();
        if (!token) {
          throw new Error("Dependency-Track BOM upload did not return a token");
        }
        return token;
      },

      async waitForBomProcessing(token: string): Promise<void> {
        for (let attempt = 0; attempt < BOM_POLL_MAX_ATTEMPTS; attempt += 1) {
          const status = await request<{ processing?: boolean }>(
            "GET",
            `/api/v1/bom/token/${encodeURIComponent(token)}`,
          );
          if (status.processing === false) {
            return;
          }
          await sleep(BOM_POLL_INTERVAL_MS);
        }
        throw new Error("Dependency-Track BOM processing timed out");
      },

      async listProjectFindings(
        projectUuid: string,
      ): Promise<DependencyTrackFinding[]> {
        return request<DependencyTrackFinding[]>(
          "GET",
          `/api/v1/finding/project/${encodeURIComponent(projectUuid)}`,
        );
      },
    };
  }
}
