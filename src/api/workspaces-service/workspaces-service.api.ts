/**
 * WorkspacesService talks to the agent-server's /api/workspaces endpoints,
 * which persist the user's saved workspaces and workspace parents on the
 * server (workspace/.openhands/workspaces.json). All clients pointed at
 * the same agent-server see the same list.
 *
 * The SDK WorkspacesClient owns compatibility preflight behavior, so old
 * agent-server backends surface the same typed version error without this
 * frontend constructing a raw HttpClient.
 *
 * `cloneRepository` is not yet in the published `@openhands/typescript-client`
 * (1.37.0); keep a local typed POST until the SDK ships it. Allowlisted in
 * `no-direct-agent-server-calls.test.ts` for that single endpoint.
 */
import axios from "axios";
import {
  WorkspacesClient,
  type WorkspacesListResponse as SdkWorkspacesListResponse,
} from "@openhands/typescript-client/clients";

import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

import {
  getAgentServerClientOptions,
  NoBackendAvailableError,
} from "../agent-server-client-options";

export interface WorkspacesListResponse {
  workspaces: LocalWorkspace[];
  workspaceParents: LocalWorkspaceParent[];
}

/** Request body for POST /api/workspaces/clone (SDK-shaped). */
export interface CloneRepositoryRequest {
  url: string;
  parentPath: string;
  providerId?: string | null;
  depth?: number | null;
}

/** Response body for POST /api/workspaces/clone (SDK-shaped). */
export interface CloneRepositoryResponse {
  path: string;
  name: string;
}

const WORKSPACES_CLONE_PATH = "/api/workspaces/clone";
const SESSION_API_KEY_HEADER = "X-Session-API-Key";
const CLONE_TIMEOUT_MS = 300_000;

function client() {
  return new WorkspacesClient(getAgentServerClientOptions());
}

function toLocalWorkspacesResponse(
  response: SdkWorkspacesListResponse,
): WorkspacesListResponse {
  return {
    workspaces: response.workspaces.map(({ parentPath, ...workspace }) => ({
      ...workspace,
      ...(parentPath ? { parentPath } : {}),
    })),
    workspaceParents: response.workspaceParents,
  };
}

class WorkspacesService {
  static async listWorkspaces(): Promise<WorkspacesListResponse> {
    return toLocalWorkspacesResponse(await client().listWorkspaces());
  }

  static async addWorkspaces(
    items: LocalWorkspace[],
  ): Promise<WorkspacesListResponse> {
    return toLocalWorkspacesResponse(await client().addWorkspaces(items));
  }

  static async removeWorkspace(path: string): Promise<void> {
    await client().deleteWorkspace(path);
  }

  static async addWorkspaceParents(
    items: LocalWorkspaceParent[],
  ): Promise<WorkspacesListResponse> {
    return toLocalWorkspacesResponse(await client().addWorkspaceParents(items));
  }

  static async removeWorkspaceParent(path: string): Promise<void> {
    await client().deleteWorkspaceParent(path);
  }

  static async cloneRepository(
    request: CloneRepositoryRequest,
  ): Promise<CloneRepositoryResponse> {
    const { host, apiKey } = getAgentServerClientOptions();
    if (!host) {
      throw new NoBackendAvailableError();
    }
    const body: Record<string, unknown> = {
      url: request.url,
      parentPath: request.parentPath,
    };
    if (request.providerId != null && request.providerId !== "") {
      body.providerId = request.providerId;
    }
    if (request.depth != null) {
      body.depth = request.depth;
    }
    const response = await axios.post<CloneRepositoryResponse>(
      `${host.replace(/\/+$/, "")}${WORKSPACES_CLONE_PATH}`,
      body,
      {
        timeout: CLONE_TIMEOUT_MS,
        headers: apiKey ? { [SESSION_API_KEY_HEADER]: apiKey } : undefined,
      },
    );
    return response.data;
  }
}

export default WorkspacesService;
