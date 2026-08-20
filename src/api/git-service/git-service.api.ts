import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { RepositoryPage, BranchPage, InstallationPage } from "#/types/git";
import { Provider } from "#/types/settings";
import { GitChange, GitChangeDiff } from "../open-hands.types";
import AgentServerConversationService from "../conversation-service/agent-server-conversation-service.api";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { mapAnyGitStatusToClientStatus } from "#/utils/git-status-mapper";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  getCloudInstallations,
  getCloudRepositoryBranches,
  searchCloudRepositories,
} from "../cloud/git-service.api";

const safeProvider = (value: string): Provider => value as Provider;

const isCloudActive = () => getActiveBackend().backend.kind === "cloud";

/**
 * Guard against null/undefined provider values that would result in
 * invalid API requests (e.g., "?provider=undefined"). Returns true
 * if the provider is falsy and the request should be skipped.
 */
const isInvalidProvider = (provider: string | null | undefined): boolean =>
  !provider || provider === "undefined" || provider === "null";

const EMPTY_REPOSITORY_PAGE: RepositoryPage = { items: [], next_page_id: null };
const EMPTY_BRANCH_PAGE: BranchPage = { items: [], next_page_id: null };
const EMPTY_INSTALLATION_PAGE: InstallationPage = {
  items: [],
  next_page_id: null,
};

class GitService {
  static async searchGitRepositories(
    query: string,
    provider: string,
    limit = 100,
    pageId?: string,
    installationId?: string,
  ): Promise<RepositoryPage> {
    if (isInvalidProvider(provider) || !isCloudActive()) {
      return EMPTY_REPOSITORY_PAGE;
    }
    return searchCloudRepositories({
      provider: safeProvider(provider),
      query: query || undefined,
      limit,
      pageId,
      installationId,
    });
  }

  static async retrieveUserGitRepositories(
    provider: string,
    pageId?: string,
    limit = 30,
    installationId?: string,
  ): Promise<RepositoryPage> {
    if (isInvalidProvider(provider) || !isCloudActive()) {
      return EMPTY_REPOSITORY_PAGE;
    }
    return searchCloudRepositories({
      provider: safeProvider(provider),
      limit,
      pageId,
      installationId,
    });
  }

  static async retrieveInstallationRepositories(
    provider: string,
    installationIndex: number,
    installations: string[],
    pageId?: string,
    limit = 30,
  ): Promise<RepositoryPage> {
    if (isInvalidProvider(provider) || !isCloudActive()) {
      return EMPTY_REPOSITORY_PAGE;
    }
    const installationId = installations[installationIndex];
    if (!installationId) return EMPTY_REPOSITORY_PAGE;
    return searchCloudRepositories({
      provider: safeProvider(provider),
      installationId,
      limit,
      pageId,
    });
  }

  static async getRepositoryBranches(
    repository: string,
    provider: string,
    query: string = "",
    pageId?: string,
    limit = 30,
  ): Promise<BranchPage> {
    if (isInvalidProvider(provider) || !isCloudActive()) {
      return EMPTY_BRANCH_PAGE;
    }
    return getCloudRepositoryBranches({
      provider: safeProvider(provider),
      repository,
      query: query || undefined,
      pageId,
      limit,
    });
  }

  static async searchRepositoryBranches(
    repository: string,
    provider: string,
    query: string,
    pageId?: string,
    limit = 30,
  ): Promise<BranchPage> {
    if (isInvalidProvider(provider) || !isCloudActive()) {
      return EMPTY_BRANCH_PAGE;
    }
    return getCloudRepositoryBranches({
      provider: safeProvider(provider),
      repository,
      query,
      pageId,
      limit,
    });
  }

  static async getUserInstallations(
    provider: string,
    pageId?: string,
    limit = 100,
  ): Promise<InstallationPage> {
    if (isInvalidProvider(provider) || !isCloudActive()) {
      return EMPTY_INSTALLATION_PAGE;
    }
    return getCloudInstallations({
      provider: safeProvider(provider),
      pageId,
      limit,
    });
  }

  static async getGitChanges(conversationId: string): Promise<GitChange[]> {
    const workingDir =
      await AgentServerConversationService.resolveConversationWorkingDir(
        conversationId,
      );
    const changes = await new RemoteWorkspace(
      getAgentServerClientOptions({ workingDir }),
    ).gitChanges(workingDir);

    return changes.map((change) => ({
      path: change.path,
      status: mapAnyGitStatusToClientStatus(
        String(change.status) as Parameters<
          typeof mapAnyGitStatusToClientStatus
        >[0],
      ),
    }));
  }

  static async getGitChangeDiff(
    _conversationId: string,
    path: string,
  ): Promise<GitChangeDiff> {
    const diff = await new RemoteWorkspace(
      getAgentServerClientOptions(),
    ).gitDiff(path);

    return {
      modified: diff.modified ?? "",
      original: diff.original ?? "",
    };
  }
}

export default GitService;
