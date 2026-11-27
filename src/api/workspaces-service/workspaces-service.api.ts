/**
 * WorkspacesService talks to the agent-server's /api/workspaces endpoints,
 * which persist the user's saved workspaces and workspace parents on the
 * server (workspace/.openhands/workspaces.json). All clients pointed at
 * the same agent-server see the same list.
 *
 * The SDK WorkspacesClient owns compatibility preflight behavior, so old
 * agent-server backends surface the same typed version error without this
 * frontend constructing a raw HttpClient.
 */
import {
  WorkspacesClient,
  type WorkspacesListResponse as SdkWorkspacesListResponse,
} from "@openhands/typescript-client/clients";

import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

import { getAgentServerClientOptions } from "../agent-server-client-options";

export interface WorkspacesListResponse {
  workspaces: LocalWorkspace[];
  workspaceParents: LocalWorkspaceParent[];
}

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
}

export default WorkspacesService;
