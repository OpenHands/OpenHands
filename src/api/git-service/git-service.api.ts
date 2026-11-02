import { GitClient } from "@openhands/typescript-client/clients";
import { RepositoryPage, BranchPage, InstallationPage } from "#/types/git";
import { Provider } from "#/types/settings";
import { getAgentServerClientOptions } from "../agent-server-client-options";
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

function normalizeRepositoryPage(page: RepositoryPage): RepositoryPage {
  return {
    items: page.items.map((repo) => ({
      ...repo,
      git_provider: safeProvider(repo.git_provider),
    })),
    next_page_id: page.next_page_id ?? null,
    ...(page.missing_token !== undefined
      ? { missing_token: page.missing_token }
      : {}),
  };
}

async function searchAgentServerRepositories(args: {
  provider: string;
  query?: string;
  limit?: number;
  pageId?: string;
}): Promise<RepositoryPage> {
  const page = await new GitClient(
    getAgentServerClientOptions(),
  ).searchRepositories(args);
  return normalizeRepositoryPage(page as RepositoryPage);
}

class GitService {
  static async searchGitRepositories(
    query: string,
    provider: string,
    limit = 100,
    pageId?: string,
    installationId?: string,
  ): Promise<RepositoryPage> {
    if (isInvalidProvider(provider)) {
      return EMPTY_REPOSITORY_PAGE;
    }
    if (!isCloudActive()) {
      return searchAgentServerRepositories({
        provider,
        query: query || undefined,
        limit,
        pageId,
      });
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
    if (isInvalidProvider(provider)) {
      return EMPTY_REPOSITORY_PAGE;
    }
    if (!isCloudActive()) {
      return searchAgentServerRepositories({
        provider,
        limit,
        pageId,
      });
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
}

export default GitService;
